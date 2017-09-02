/* yapio - "Yet another parallel I/O (test)".
 *
 * yapio is an MPI program useful for benchmarking parallel I/O systems in
 * addition to verifying file data integrity and consistency across distributed
 * tasks.
 *
 * Written by Paul Nowoczynski.
 */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <getopt.h>
#include <mpi.h>
#include <errno.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define YAPIO_OPTS "b:n:hd:p:k"

#define YAPIO_DEF_NBLKS_PER_PE     1000
#define YAPIO_DEF_BLK_SIZE         4096
#define YAPIO_MAX_BLK_SIZE         (1ULL << 30)
#define YAPIO_MAX_SIZE_PER_PE      (1ULL << 40)
#define YAPIO_DEFAULT_FILE_PREFIX  "yapio."
#define YAPIO_MKSTEMP_TEMPLATE     "XXXXXX"
#define YAPIO_MKSTEMP_TEMPLATE_LEN 6

enum yapio_log_levels
{
    YAPIO_LL_FATAL = 0,
    YAPIO_LL_ERROR = 1,
    YAPIO_LL_WARN  = 2,
    YAPIO_LL_DEBUG = 3,
    YAPIO_LL_TRACE = 4,
    YAPIO_LL_MAX
};

enum yapio_patterns
{
    YAPIO_IOP_SEQUENTIAL,
    YAPIO_IOP_RANDOM,
    YAPIO_IOP_STRIDED,
};

#define YAPIO_EXIT_OK  0
#define YAPIO_EXIT_ERR 1

static size_t      yapioNumBlksPerRank = YAPIO_DEF_NBLKS_PER_PE;
static size_t      yapioBlkSz          = YAPIO_DEF_BLK_SIZE;
static char       *yapioFilePrefix     = YAPIO_DEFAULT_FILE_PREFIX;
static int         yapioDbgLevel       = YAPIO_LL_WARN;
static bool        yapioMpiInit        = false;
static bool        yapioKeepFile       = false;
static const char *yapioExecName;
static const char *yapioTestRootDir;
static char        yapioTestFileName[PATH_MAX + 1];
static int         yapioMyRank;
static int         yapioNumRanks;
static int         yapioFileDesc;
static int         yapioTestIteration;
static char       *yapioIOBuf;

/* Set of magic numbers which will be used for block tagging.
 */
#define YAPIO_NUM_BLK_MAGICS 4
static const unsigned long long yapioBlkMagics[YAPIO_NUM_BLK_MAGICS] =
{0xa3cfad825d, 0xf0f0f0f0f0f0f0f0, 0x181ce41215, 0x01030507090a0c0e};

static unsigned long long
yapio_get_blk_magic(size_t blk_num)
{
    return yapioBlkMagics[blk_num % YAPIO_NUM_BLK_MAGICS];
}

typedef struct yapio_blk_metadata
{
    int    ybm_writer_rank;     //rank who has last written this block
    int    ybm_write_iteration; //iteration number of last write
    size_t ybm_blk_number;      //number of the block
} yapio_blk_md_t;

yapio_blk_md_t  *yapioSourceBlkMd; //metadata which this rank maintains

typedef struct yapio_test_context
{
    bool                ytc_leave_holes;      //some writes are skipped
    bool                ytc_backwards;        //IOs are done in reverse order
    bool                ytc_skip_mpi_gather;  //tests local buffers only
//    bool                ytc_no_fsync;
    bool                ytc_write;
    enum yapio_patterns ytc_io_pattern;       //IO pattern to be employed
    yapio_blk_md_t    **ytc_ops_md;           //operations metadata per block
    int                *ytc_num_ops_per_rank; //count for ytc_ops_md arrays
} yapio_test_ctx_t;

static const char *
yapio_ll_to_string(enum yapio_log_levels yapio_ll)
{
    switch (yapio_ll)
    {
    case YAPIO_LL_FATAL:
        return "fatal";
    case YAPIO_LL_ERROR:
        return "error";
    case YAPIO_LL_WARN:
        return "warn";
    case YAPIO_LL_TRACE:
        return "trace";
    default:
        break;
    }

    return "debug";
}

static void
yapio_exit(int exit_rc)
{
    if (yapioMpiInit)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    exit(exit_rc);
}

#define log_msg(lvl, message, ...)                                  \
    {                                                               \
        if (lvl <= yapioDbgLevel)                                   \
        {                                                           \
            fprintf(stderr, "<%s:%s:%d@%u> " message "\n",          \
                    yapio_ll_to_string(lvl), __func__, yapioMyRank, \
                    __LINE__, ##__VA_ARGS__);                       \
            if (lvl == YAPIO_LL_FATAL)                              \
                yapio_exit(YAPIO_EXIT_ERR);                         \
        }                                                           \
    }

static void
yapio_print_help(int exit_val)
{
    fprintf(exit_val ? stderr : stdout,
            "%s [OPTION] DIRECTORY\n\n"
            "Options:\n"
            "\t-b    block size\n"
            "\t-d    debugging level\n"
            "\t-h    print help message\n"
            "\t-k    keep file after test completion\n"
            "\t-n    number of blocks per task\n"
            "\t-p    filename prefix\n",
            yapioExecName);

    exit(exit_val);
}

static void
yapio_mpi_setup(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &yapioMyRank);
    MPI_Comm_size(MPI_COMM_WORLD, &yapioNumRanks);

    yapioMpiInit = true;
}

static bool
yapio_leader_rank(void)
{
    return yapioMyRank == 0 ? true : false;
}

static void
yapio_getopts(int argc, char **argv)
{
    int opt;

    yapioExecName = argv[0];

    while ((opt = getopt(argc, argv, YAPIO_OPTS)) != -1)
    {
        switch (opt)
        {
        case 'b':
            yapioBlkSz = strtoull(optarg, NULL, 10);
            break;
        case 'd':
            yapioDbgLevel = MIN(atoi(optarg), YAPIO_LL_MAX);
            break;
        case 'h':
            yapio_print_help(YAPIO_EXIT_OK);
            break;
        case 'k':
            yapioKeepFile = true;
            break;
        case 'n':
            yapioNumBlksPerRank = strtoull(optarg, NULL, 10);
            break;
        case 'p':
            yapioFilePrefix = optarg;
            break;
        default:
            yapio_print_help(YAPIO_EXIT_ERR);
            break;
        }
    }

    /* Check for user provided test directory parameter which should be at the
     * end.
     */
    yapioTestRootDir = argv[optind];
    if (!yapioTestRootDir || argc > (optind + 1))
        yapio_print_help(YAPIO_EXIT_ERR);

    if ((yapioNumBlksPerRank * yapioBlkSz) > YAPIO_MAX_SIZE_PER_PE)
        log_msg(YAPIO_LL_FATAL,
                "Per rank data size (%zu) exceeds max (%llu)",
                (yapioNumBlksPerRank * yapioBlkSz), YAPIO_MAX_SIZE_PER_PE);

    if (yapio_leader_rank())
    {
        log_msg(YAPIO_LL_DEBUG, "nblks=%zu blksz=%zu",
                yapioNumBlksPerRank, yapioBlkSz);
        log_msg(YAPIO_LL_DEBUG, "prefix=%s dirname=%s",
                yapioFilePrefix, yapioTestRootDir);
        log_msg(YAPIO_LL_DEBUG, "rank=%d num_ranks=%d",
                yapioMyRank, yapioNumRanks);
    }
}

/**
 * yapio_verify_test_directory - ensure test directory exists.
 */
static void
yapio_verify_test_directory(void)
{
    struct stat stb;
    int rc = stat(yapioTestRootDir, &stb);
    if (rc)
        rc = errno;

    else if (!S_ISDIR(stb.st_mode))
        rc = ENOTDIR;

    if (rc)
        log_msg(YAPIO_LL_FATAL, "%s", strerror(rc));
}

/**
 * yapio_setup_test_file - Rank0 will create a temp file and broadcast
 *    the name the to the other ranks who will then also open the temp file.
 */
static void
yapio_setup_test_file(void)
{
    yapio_verify_test_directory();

    int path_len = snprintf(yapioTestFileName, PATH_MAX, "%s/%s%s",
                            yapioTestRootDir, yapioFilePrefix,
                            YAPIO_MKSTEMP_TEMPLATE);
    if (path_len > PATH_MAX)
        log_msg(YAPIO_LL_FATAL, "%s", strerror(ENAMETOOLONG));

    if (yapio_leader_rank())
    {
        yapioFileDesc = mkstemp(yapioTestFileName);
        if (yapioFileDesc < 0)
            log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));

        log_msg(YAPIO_LL_DEBUG, "%s", yapioTestFileName);
    }

    /* Broadcast only the section of the filename which was modified by
     * mkstemp().
     */
    MPI_Bcast(&yapioTestFileName[path_len - YAPIO_MKSTEMP_TEMPLATE_LEN],
              YAPIO_MKSTEMP_TEMPLATE_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (!yapio_leader_rank())
    {
        yapioFileDesc = open(yapioTestFileName, O_RDWR);
        if (yapioFileDesc < 0)
            log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));

    }
}

static void
yapio_close_test_file(void)
{
    int rc = close(yapioFileDesc);
    if (rc < 0)
        log_msg(YAPIO_LL_FATAL, "close: %s", strerror(errno));
}

static void
yapio_destroy_buffers(void)
{
    if (yapioSourceBlkMd)
        free(yapioSourceBlkMd);

    if (yapioIOBuf)
        free(yapioIOBuf);
}

static void
yapio_destroy_buffers_and_abort(void)
{
    yapio_destroy_buffers();
    log_msg(YAPIO_LL_FATAL, "yapio_setup_buffers() %s", strerror(ENOMEM));
}

static void
yapio_alloc_buffers(void)
{
    yapioSourceBlkMd = calloc(yapioNumBlksPerRank, sizeof(yapio_blk_md_t));
    if (yapioSourceBlkMd == NULL)
        yapio_destroy_buffers_and_abort();

    yapioIOBuf = calloc(1, yapioBlkSz);
    if (yapioIOBuf == NULL)
        yapio_destroy_buffers_and_abort();
}

static void
yapio_initialize_source_md_buffer(void)
{
    int i;
    for (i = 0; i < yapioNumBlksPerRank; i++)
    {
        yapioSourceBlkMd[i].ybm_writer_rank = yapioMyRank;
        yapioSourceBlkMd[i].ybm_blk_number =
            (yapioMyRank * yapioNumBlksPerRank) + i;
    }
}

static unsigned long long
yapio_get_content_word(const yapio_blk_md_t *md, size_t word_num)
{
    unsigned long long content_word =
        (yapio_get_blk_magic(md->ybm_blk_number) + md->ybm_writer_rank +
         md->ybm_blk_number + word_num);

    return content_word;
}

static void
yapio_apply_contents_to_io_buffer(char *buf, size_t buf_len,
                                  const yapio_blk_md_t *md)
{
    unsigned long long *buffer_of_longs = (unsigned long long *)buf;
    size_t num_words = buf_len / sizeof(unsigned long long);

    size_t i;
    for (i = 0; i < num_words; i++)
    {
        buffer_of_longs[i] = yapio_get_content_word(md, i);

        log_msg(YAPIO_LL_TRACE, "%zu:%llx", i, buffer_of_longs[i]);
    }
}

static int
yapio_verify_contents_of_io_buffer(const char *buf, size_t buf_len,
                                   const yapio_blk_md_t *md)
{
    const unsigned long long *buffer_of_longs = (unsigned long long *)buf;
    size_t num_words = buf_len / sizeof(unsigned long long);

    size_t i;
    for (i = 0; i < num_words; i++)
    {
        if (buffer_of_longs[i] != yapio_get_content_word(md, i))
        {
            log_msg(YAPIO_LL_ERROR, "blk=%zu word=%zu got=%llx expected=%llx",
                    md->ybm_blk_number, i, buffer_of_longs[i],
                    yapio_get_content_word(md, i));

            return -1;
        }

        log_msg(YAPIO_LL_TRACE, "OK %zu:%llx", i, buffer_of_longs[i]);
    }

    return 0;
}

static off_t
yapio_get_rw_offset(const yapio_blk_md_t *md, size_t blk_sz)
{
    off_t rw_offset = md->ybm_blk_number * blk_sz;

    return rw_offset;
}

/**
 * yapio_initialize_test_file_contents - each rank in the job writes its
 *    implicitly assigned set of contiguous blocks.
 */
static int
yapio_perform_io(yapio_test_ctx_t *ytc)
{
    int rc = 0;
    int i;

    MPI_Barrier(MPI_COMM_WORLD);

    for (i = 0; i < yapioNumRanks; i++)
    {
        int rank_idx = ytc->ytc_backwards ? (yapioNumRanks - i - 1) : i;

        const yapio_blk_md_t *md_array = ytc->ytc_ops_md[rank_idx];
        if (!md_array)
            continue;

        int j;
        for (j = 0; j < ytc->ytc_num_ops_per_rank[rank_idx]; j++)
        {
            const yapio_blk_md_t *md = &md_array[j];

            if (ytc->ytc_write)
                yapio_apply_contents_to_io_buffer(yapioIOBuf, yapioBlkSz, md);

            /* Obtain this IO's offset from the
             */
            off_t off = yapio_get_rw_offset(md, yapioBlkSz);
            if (off < 0)
            {
                log_msg(YAPIO_LL_ERROR, "yapio_get_rw_offset() failed");
                rc = -ERANGE;
                break;
            }

            ssize_t io_rc, io_bytes = 0;
            do
            {
                char *adjusted_buf = yapioIOBuf + io_bytes;
                size_t adjusted_io_len = yapioBlkSz - io_bytes;
                off_t adjusted_off = off + io_bytes;

                io_rc = ytc->ytc_write ?
                    pwrite(yapioFileDesc, adjusted_buf, adjusted_io_len,
                           adjusted_off) :
                    pread(yapioFileDesc, adjusted_buf, adjusted_io_len,
                          adjusted_off);

                log_msg(YAPIO_LL_DEBUG, "%s rc=%zd %lu",
                        ytc->ytc_write ? "pwrite" : "pread", io_rc,
                        adjusted_off);

                if (io_rc > 0)
                    io_bytes += io_rc;

            } while (io_rc > 0 && io_bytes < yapioBlkSz);

            if (io_rc < 0)
            {
                log_msg(YAPIO_LL_ERROR, "io failed at offset %lu: %s",
                        off, strerror(errno));
                rc = -errno;
                break;
            }

            if (!ytc->ytc_write)
            {
                rc = yapio_verify_contents_of_io_buffer(yapioIOBuf, yapioBlkSz,
                                                        md);
                if (rc)
                    break;
            }
        }
    }

    if (!rc)
    {
        rc = fsync(yapioFileDesc);
        if (rc)
        {
            rc = -errno;
            log_msg(YAPIO_LL_ERROR, "fsync(): %s", strerror(errno));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* Increment the test iteration
     */
    yapioTestIteration++;

    return rc;
}

static void
yapio_setup_buffers(void)
{
    yapio_alloc_buffers();
    yapio_initialize_source_md_buffer();
}

static void
yapio_unlink_test_file(void)
{
    if (!yapioKeepFile && yapio_leader_rank())
    {
        int rc = unlink(yapioTestFileName);
        if (rc)
        {
            log_msg(YAPIO_LL_ERROR, "unlink %s: %s", yapioTestFileName,
                    strerror(errno));

            yapio_exit(YAPIO_EXIT_ERR);
        }

        log_msg(YAPIO_LL_DEBUG, "%s", yapioTestFileName);
    }
}

static void
yapio_test_ctx_release(yapio_test_ctx_t *ytc)
{
    if (ytc->ytc_ops_md)
    {
        int i;
        for (i = 0; i < yapioNumRanks; i++)
            if (ytc->ytc_ops_md[i])
                free(ytc->ytc_ops_md[i]);

        free(ytc->ytc_ops_md);
    }

    if (ytc->ytc_num_ops_per_rank)
        free(ytc->ytc_num_ops_per_rank);
}

static int
yapio_read_from_dev_urandom(void *buffer, size_t size)
{
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd < 0)
    {
        fd = -errno;
        log_msg(YAPIO_LL_ERROR, "open: %s", strerror(errno));
        return fd;
    }

    int error = 0;
    size_t nbytes = 0;
    do
    {
        ssize_t rc = read(fd, ((char *)buffer) + nbytes, size - nbytes);
        if (rc < 0)
        {
            error = -errno;
            log_msg(YAPIO_LL_ERROR, "read: %s", strerror(errno));
            break;
        }

        nbytes += rc;

    } while (nbytes < size);

    close(fd);
    return error;
}

/**
 * yapio_test_context_setup_skip_mpi_gather - setup driver for the 'local' I/O
 *    mode where each rank determines its own work queue.
 * Note: YAPIO_IOP_STRIDED mode is not supported in this mode.
 */
static int
yapio_test_context_setup_skip_mpi_gather(yapio_test_ctx_t *ytc)
{
    if (!ytc->ytc_skip_mpi_gather)
        return -EINVAL;

    if (ytc->ytc_io_pattern == YAPIO_IOP_STRIDED)
        return -ENOTSUP;

    /* In this mode all operation instructions are pulled from the local
     * Source Blk Metadata - there is no exchange with other ranks.
     */
    ytc->ytc_num_ops_per_rank[yapioMyRank] = yapioNumBlksPerRank;
    ytc->ytc_ops_md[yapioMyRank] = calloc(yapioNumBlksPerRank,
                                          sizeof(yapio_blk_md_t));

    yapio_blk_md_t *md = ytc->ytc_ops_md[yapioMyRank];
    if (!md)
        return -ENOMEM;

    if (ytc->ytc_io_pattern == YAPIO_IOP_SEQUENTIAL)
    {
        int i;
        for (i = 0; i < yapioNumBlksPerRank; i++)
        {
            int src_idx = i;

            if (ytc->ytc_backwards)
                src_idx = (yapioNumBlksPerRank - 1 - i);

            md[i] = yapioSourceBlkMd[src_idx];
            log_msg(YAPIO_LL_TRACE, "rank=%d md->ybm_blk_number=%zd",
                    md[i].ybm_writer_rank, md[i].ybm_blk_number);
        }
    }
    else if (ytc->ytc_io_pattern == YAPIO_IOP_RANDOM)
    {
        size_t buf_sz = yapioNumBlksPerRank * sizeof(int);
        int *array_of_randoms = malloc(buf_sz);
        if (!array_of_randoms)
            return -ENOMEM;

        int rc = yapio_read_from_dev_urandom((void *)array_of_randoms, buf_sz);
        if (rc)
            return rc;

        int i;
        for (i = 0; i < yapioNumBlksPerRank; i++)
            md[i] = yapioSourceBlkMd[i];

        for (i = 0; i < yapioNumBlksPerRank; i++)
        {
            int swap_idx = array_of_randoms[i] % yapioNumBlksPerRank;
            yapio_blk_md_t md_tmp = md[swap_idx];
            md[swap_idx] = md[i];
            md[i] = md_tmp;

            log_msg(YAPIO_LL_TRACE, "swapped %d:%zu <-> %d:%zu",
                    i, md[swap_idx].ybm_blk_number, swap_idx,
                    md[i].ybm_blk_number);
        }
    }
    else
    {
        log_msg(YAPIO_LL_WARN, "unknown io_pattern=%d", ytc->ytc_io_pattern);
        return -EINVAL;
    }

    return 0;
}

/**
 * yapio_test_context_setup - Allocates memory buffers for the test according
 *   to the input data in the yapio_test_ctx_t.  After memory allocation, this
 *   function will arrange the yapio_blk_md_t either autonomously or in
 *   conjunction with the other ranks in the job.  On completion, the 'ytc'
 *   will have an array of pointers stored in ytc->ytc_ops_md which hold the
 *   set of operations to be performed by this rank.
 */
static int
yapio_test_context_setup(yapio_test_ctx_t *ytc)
{
    ytc->ytc_ops_md = calloc(yapioNumRanks, sizeof(yapio_blk_md_t *));
    if (!ytc->ytc_ops_md)
        return -ENOMEM;

    ytc->ytc_num_ops_per_rank = calloc(yapioNumRanks, sizeof(int));
    if (!ytc->ytc_num_ops_per_rank)
    {
        yapio_test_ctx_release(ytc);
        return -ENOMEM;
    }

    if (ytc->ytc_skip_mpi_gather)
    {
        int rc = yapio_test_context_setup_skip_mpi_gather(ytc);
        if (rc)
        {
            log_msg(YAPIO_LL_ERROR,
                    "yapio_test_context_setup_skip_mpi_gather: %s",
                    strerror(-rc));

            yapio_test_ctx_release(ytc);

            return rc;
        }
    }

    return 0;
}

#define YAPIO_SIMPLE_IO_TEST(ctx, write)                            \
    {                                                               \
        (ctx)->ytc_leave_holes     = false;                         \
        (ctx)->ytc_backwards       = false;                         \
        (ctx)->ytc_skip_mpi_gather = true;                          \
        (ctx)->ytc_write           = write;                         \
        (ctx)->ytc_io_pattern      = YAPIO_IOP_SEQUENTIAL;          \
    }

int
main(int argc, char *argv[])
{
    yapio_test_ctx_t my_test_ctx;

    yapio_mpi_setup(argc, argv);
    yapio_getopts(argc, argv);

    yapio_setup_buffers();

    yapio_setup_test_file();

    YAPIO_SIMPLE_IO_TEST(&my_test_ctx, true);
    my_test_ctx.ytc_io_pattern = YAPIO_IOP_RANDOM;
//    my_test_ctx.ytc_backwards = true;
    int rc = yapio_test_context_setup(&my_test_ctx);
    if (rc)
        yapio_exit(YAPIO_EXIT_ERR);

    yapio_perform_io(&my_test_ctx);

    yapio_test_ctx_release(&my_test_ctx);

    YAPIO_SIMPLE_IO_TEST(&my_test_ctx, false);
    my_test_ctx.ytc_io_pattern = YAPIO_IOP_RANDOM;
//    my_test_ctx.ytc_backwards = true;
    rc = yapio_test_context_setup(&my_test_ctx);
    if (rc)
        yapio_exit(YAPIO_EXIT_ERR);

    yapio_perform_io(&my_test_ctx);
    yapio_test_ctx_release(&my_test_ctx);

    yapio_close_test_file();

    yapio_destroy_buffers();

    yapio_unlink_test_file();

    yapio_exit(YAPIO_EXIT_OK);

    return 0;
}
