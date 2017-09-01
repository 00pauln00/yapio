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

#define YAPIO_OPTS "b:n:hd:p:"

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
    YAPIO_LL_MAX
};

#define YAPIO_EXIT_OK  0
#define YAPIO_EXIT_ERR 1

static size_t      yapioNumBlksPerRank = YAPIO_DEF_NBLKS_PER_PE;
static size_t      yapioBlkSz          = YAPIO_DEF_BLK_SIZE;
static char       *yapioFilePrefix     = YAPIO_DEFAULT_FILE_PREFIX;
static int         yapioDbgLevel       = YAPIO_LL_WARN;
static bool        yapioMpiInit        = false;
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

yapio_blk_md_t *yapioSourceBlkMd; //metadata which this rank maintains
yapio_blk_md_t *yapioSinkBlkMd;   //metadata pushed from others to this rank

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
    if (yapioSinkBlkMd)
        free(yapioSinkBlkMd);
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

    yapioSinkBlkMd = calloc(yapioNumBlksPerRank, sizeof(yapio_blk_md_t));
    if (yapioSinkBlkMd == NULL)
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

        yapioSinkBlkMd[i].ybm_writer_rank = yapioMyRank;
        yapioSinkBlkMd[i].ybm_blk_number =
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

        log_msg(YAPIO_LL_DEBUG, "%zu:%llx", i, buffer_of_longs[i]);
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

        log_msg(YAPIO_LL_DEBUG, "OK %zu:%llx", i, buffer_of_longs[i]);
    }

    return 0;
}

static off_t
yapio_get_rw_offset(size_t op_num)
{
    if (op_num >= yapioNumBlksPerRank)
        return -1;

    off_t rw_offset = yapioSinkBlkMd[op_num].ybm_blk_number * yapioBlkSz;

    return rw_offset;
}

/**
 * yapio_initialize_test_file_contents - each rank in the job writes its
 *    implicitly assigned set of contiguous blocks.
 */
static int
yapio_rw(bool write)
{
    int rc = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    size_t i;
    for (i = 0; i < yapioNumBlksPerRank; i++)
    {
        const yapio_blk_md_t *md = &yapioSinkBlkMd[i];

        if (write)
            yapio_apply_contents_to_io_buffer(yapioIOBuf, yapioBlkSz, md);

        /* Obtain this IO's offset from the
         */
        off_t off = yapio_get_rw_offset(i);
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

            io_rc = write ?
                pwrite(yapioFileDesc, adjusted_buf, adjusted_io_len,
                       adjusted_off) :
                pread(yapioFileDesc, adjusted_buf, adjusted_io_len,
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

        if (!write)
        {
            rc =
                yapio_verify_contents_of_io_buffer(yapioIOBuf, yapioBlkSz, md);
            if (rc)
                break;
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

int
main(int argc, char *argv[])
{
    yapio_mpi_setup(argc, argv);
    yapio_getopts(argc, argv);

    yapio_setup_buffers();

    yapio_setup_test_file();

    yapio_rw(true);
    yapio_rw(false);

    yapio_close_test_file();

    yapio_destroy_buffers();

    yapio_exit(YAPIO_EXIT_OK);

    return 0;
}
