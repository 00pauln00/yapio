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
#include <time.h>
#include <pthread.h>

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define YAPIO_OPTS "b:n:hd:p:kt:PD:sF"

#define YAPIO_DEF_NBLKS_PER_PE     1000
#define YAPIO_DEF_BLK_SIZE         4096
#define YAPIO_MAX_BLK_SIZE         (1ULL << 30)
#define YAPIO_MAX_SIZE_PER_PE      (1ULL << 40)
#define YAPIO_DEFAULT_FILE_PREFIX  "yapio."
#define YAPIO_MKSTEMP_TEMPLATE     "XXXXXX"
#define YAPIO_MKSTEMP_TEMPLATE_LEN 6
#define YAPIO_DECOMPOSE_MAX        8

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
    YAPIO_IOP_MAX
};

#define YAPIO_EXIT_OK  0
#define YAPIO_EXIT_ERR 1

static size_t      yapioNumBlksPerRank = YAPIO_DEF_NBLKS_PER_PE;
static size_t      yapioBlkSz          = YAPIO_DEF_BLK_SIZE;
static char       *yapioFilePrefix     = YAPIO_DEFAULT_FILE_PREFIX;
static int         yapioDbgLevel       = YAPIO_LL_WARN;
static bool        yapioMpiInit        = false;
static bool        yapioKeepFile       = false;
static bool        yapioFilePerProcess = false;
static bool        yapioPolluteBlks    = false;
static bool        yapioDisplayStats   = false;
static int         yapioDecomposeCnt   = 0;
static const char *yapioExecName;
static const char *yapioTestRootDir;
static char        yapioTestFileName[PATH_MAX + 1];
static char        yapioTestFileNameFpp[PATH_MAX + 1];
static int         yapioMyRank;
static int         yapioNumRanks;
static int         yapioFileDesc;
static int        *yapioFileDescFpp;
static char       *yapioIOBuf;

static pthread_t       yapioStatsCollectionThread;
static pthread_t       yapioStatsReportingThread;
static pthread_cond_t  yapioThreadCond  = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t yapioThreadMutex = PTHREAD_MUTEX_INITIALIZER;

#define YAPIO_STATS_SLEEP_USEC 10000

#define MPI_OP_START pthread_mutex_lock(&yapioThreadMutex)
#define MPI_OP_END   pthread_mutex_unlock(&yapioThreadMutex)

static int
yapio_mpi_barrier(MPI_Comm comm)
{
    int rc;

    MPI_OP_START;
    rc = MPI_Barrier(comm);
    MPI_OP_END;

    return rc;
}

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
    int    ybm_owner_rank_fpp;  //fpp only, file to which contents belong
    size_t ybm_blk_number;      //number of the block - should not change!
} yapio_blk_md_t;

yapio_blk_md_t  *yapioSourceBlkMd; //metadata which this rank maintains

typedef struct yapio_test_context_md_handle
{
    yapio_blk_md_t *ytcmh_ops;
    int             ytcmh_num_ops;
} yapio_test_ctx_md_t;

typedef struct timespec yapio_timer_t;

enum yapio_test_ctx_mdh_in_out
{
    YAPIO_TEST_CTX_MDH_IN  = 0,
    YAPIO_TEST_CTX_MDH_OUT = 1,
    YAPIO_TEST_CTX_MDH_MAX = 2,
};

enum yapio_test_ctx_run
{
    YAPIO_TEST_CTX_RUN_NOT_STARTED,
    YAPIO_TEST_CTX_RUN_STARTED,
    YAPIO_TEST_CTX_RUN_COMPLETE,
    YAPIO_TEST_CTX_RUN_STATS_REPORTED,
};

enum yapio_barrier_stats
{
    YAPIO_BARRIER_STATS_AVG  = 0,
    YAPIO_BARRIER_STATS_MAX  = 1,
    YAPIO_BARRIER_STATS_MED  = 2,
    YAPIO_BARRIER_STATS_LAST = 3
};

struct yapio_test_group;

typedef struct yapio_test_context
{
    unsigned                 ytc_leave_holes:1,
                             ytc_backwards:1,
                             ytc_remote_locality:1,
                             ytc_read:1,
                             ytc_no_fsync:1;
    enum yapio_patterns      ytc_io_pattern;       //IO pattern to be employed
    int                      ytc_test_num;
    enum yapio_test_ctx_run  ytc_run_status;
    yapio_timer_t            ytc_setup_time;
    yapio_timer_t            ytc_test_duration;
    yapio_timer_t            ytc_reported_time;
    yapio_timer_t            ytc_barrier_wait[2];
    yapio_test_ctx_md_t      ytc_in_out_md_ops[YAPIO_TEST_CTX_MDH_MAX];
    float                    ytc_barrier_results[YAPIO_BARRIER_STATS_LAST];
    int                      ytc_barrier_max_rank;
    struct yapio_test_group *ytc_group;
} yapio_test_ctx_t;

#define YAPIO_NUM_TEST_CTXS_MAX 256

typedef struct yapio_test_group
{
    MPI_Comm         ytg_comm;
    MPI_Group        ytg_group;
    yapio_test_ctx_t ytg_contexts[YAPIO_NUM_TEST_CTXS_MAX];
    int              ytg_num_contexts;
    int              ytg_group_num;
    int              ytg_first_rank;
    int              ytg_num_ranks;
    size_t           ytg_num_blks_per_rank;
    size_t           ytg_blk_sz;
    bool             ytg_file_per_process;
    bool             ytg_leader_rank;
} yapio_test_group_t;

static yapio_test_group_t *yapioMyTestGroup;
static yapio_test_group_t  yapioTestGroups[YAPIO_NUM_TEST_CTXS_MAX];
static int                 yapioNumTestGroups;

static int
yapio_relative_rank_get(const yapio_test_group_t *ytg, int idx_shift)
{
    return
        (yapioMyRank - ytg->ytg_first_rank + idx_shift) % ytg->ytg_num_ranks;
}

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

static bool
yapio_global_leader_rank(void)
{
    return !yapioMyRank ? true : false;
}

static bool
yapio_leader_rank(void)
{
    return yapioMyTestGroup ? yapioMyTestGroup->ytg_leader_rank :
        yapio_global_leader_rank();
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

#define log_msg_r0(lvl, message, ...)       \
    {                                           \
        if (yapio_leader_rank())                \
            log_msg(lvl, message, ##__VA_ARGS__);   \
    }

static void
yapio_exit(int exit_rc)
{
    if (yapioMpiInit)
    {
        yapio_mpi_barrier(MPI_COMM_WORLD);
        MPI_Finalize();
    }

    exit(exit_rc);
}

/* Timer related calls.
 */
#define YAPIO_NSEC_PER_SEC 1000000000L
#define YAPIO_USEC_PER_SEC 1000000L

#define YAPIO_TIME_PRINT_SPEC "%.4f"
#define YAPIO_TIMER_ARGS(timer)                                         \
    (float)((timer)->tv_sec +                                           \
            (float)((float)(timer)->tv_nsec / YAPIO_NSEC_PER_SEC))

/**
 * yapio_get_time - wrapper for clock_gettime()
 */
static void
yapio_get_time(yapio_timer_t *yt)
{
    if (clock_gettime(CLOCK_MONOTONIC_RAW, yt))
        log_msg(YAPIO_LL_FATAL, "clock_gettime: %s", strerror(errno));
}

/**
 * yapio_start_timer - starts a 'timer'.
 */
static void
yapio_start_timer(yapio_timer_t *timer)
{
    yapio_get_time(timer);
}
/**
 * yapio_get_time_duration - take end timestamp and subtract it from the value
 *    in 'timer', leaving the elapsed time.
 */
static void
yapio_get_time_duration(yapio_timer_t *result, const yapio_timer_t *end)
{
    result->tv_sec = end->tv_sec - result->tv_sec;
    result->tv_nsec = end->tv_nsec - result->tv_nsec;

    if (result->tv_nsec < 0)
    {
        result->tv_sec--;
        result->tv_nsec += YAPIO_NSEC_PER_SEC;
    }

    if (result->tv_nsec < 0 || result->tv_nsec >= YAPIO_NSEC_PER_SEC)
        result->tv_nsec = 0;
}

static void
yapio_end_timer(yapio_timer_t *result)
{
    yapio_timer_t end_timer;
    yapio_start_timer(&end_timer);
    yapio_get_time_duration(result, &end_timer);
}

static void
yapio_print_help(int exit_val)
{
    if (yapio_global_leader_rank())
        fprintf(exit_val ? stderr : stdout,
                "%s [OPTION] DIRECTORY\n\n"
                "Options:\n"
                "\t-b  Block size\n"
                "\t-d  Debugging level\n"
                "\t-F  File per process\n"
                "\t-h  Print help message\n"
                "\t-k  Keep file after test completion\n"
                "\t-n  Number of blocks per task\n"
                "\t-p  File name prefix\n"
                "\t-s  Display test duration and barrier wait times\n"
                "\t-t  Test description\n"
                "\t    - Pattern:    sequential (s), random (R), strided (S)\n"
                "\t    - I/O Op:     read (r), write (w)\n"
                "\t    - Locality:   local (L), distributed (D)\n"
                "\t    - Options:    backwards (b), holes (h), no-fsync (f)\n"
                "\t    - Parameters: block size (B), blocks per rank (n),\n"
                "\t                  num ranks (N), file-per-process (F)\n"
                "\n\t    Example: -t wsL,rRD\n"
                "\t      sequential write, distribute reads randomly\n"
                "\n\t    Example: -t wsL,rRD -t N4:B4096:F:ws,rsb\n"
                "\t      Run two tests simultaneously\n",
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

#define YAPIO_RECIPE_PARAM_STRLEN_MAX 17

static int
yapio_test_recipe_param_to_ull(const char *param_str, size_t *result)
{
    char recipe_param_tmp_str[YAPIO_RECIPE_PARAM_STRLEN_MAX];
    int j;

    for (j = 0; j < YAPIO_RECIPE_PARAM_STRLEN_MAX - 1 && param_str[j] != ':';
         j++)
        recipe_param_tmp_str[j] = param_str[j];

    if (param_str[j] != ':')
    {
        if (j == YAPIO_RECIPE_PARAM_STRLEN_MAX)
        {
            log_msg(YAPIO_LL_ERROR, "recipe input string is too long");
            return -E2BIG;
        }
        return -EINVAL;
    }

    recipe_param_tmp_str[j] = '\0';

    if (result)
        *result = strtoull(recipe_param_tmp_str, NULL, 10);

    log_msg_r0(YAPIO_LL_TRACE, "%s %zu", recipe_param_tmp_str, *result);

    return j + 1; //include ':' char
}

static void
yapio_test_group_init(yapio_test_group_t *ytg)
{
    ytg->ytg_num_contexts = 1;
    ytg->ytg_num_blks_per_rank = yapioNumBlksPerRank;
    ytg->ytg_blk_sz = yapioBlkSz;
    ytg->ytg_file_per_process = yapioFilePerProcess;
    ytg->ytg_group_num = yapioNumTestGroups;
}

#define YAPIO_RECIPE_STRLEN_MAX 4097
static int
yapio_parse_test_recipe(const char *recipe_str)
{
    yapio_test_group_t *ytg = &yapioTestGroups[yapioNumTestGroups];

    yapio_test_group_init(ytg);

    size_t recipe_str_len = strnlen(recipe_str, YAPIO_RECIPE_STRLEN_MAX);
    if (recipe_str_len >= YAPIO_RECIPE_STRLEN_MAX)
    {
        log_msg(YAPIO_LL_ERROR, "recipe input string is too long");
        return -E2BIG;
    }

    int rc = 0;
    int rw, locality, io_pattern;
    int test_ctx_idx;
    size_t tmp;

    rw = locality = io_pattern = -1;

    unsigned i;
    for (i = 0; i < recipe_str_len; i++)
    {
        test_ctx_idx = ytg->ytg_num_contexts - 1;
        yapio_test_ctx_t *ytc = &ytg->ytg_contexts[test_ctx_idx];

        /* If mutually exclusive options have been violated then exit the loop.
         */
        if (rw > 0 || locality > 0 || io_pattern > 0)
            break;

        const char c = recipe_str[i];
        switch (c)
        {
        /* Recipe specific parameters (B,n,N,F) which are separated by ':'
         */
        case 'B':
            i += yapio_test_recipe_param_to_ull(&recipe_str[i + 1],
                                                &ytg->ytg_blk_sz);
            break;
        case 'n':
            i += yapio_test_recipe_param_to_ull(&recipe_str[i + 1],
                                                &ytg->ytg_num_blks_per_rank);
            break;
        case 'N':
            i += yapio_test_recipe_param_to_ull(&recipe_str[i + 1], &tmp);
            ytg->ytg_num_ranks = (int)tmp;
            break;
        case 'F':
            ytg->ytg_file_per_process = true;
            i++;
            break;

        case 'w': //'w' and 'r' are mutually exclusive
            if (!++rw)
                ytc->ytc_read = 0;
            break;

        case 'r':
            if (!++rw)
                ytc->ytc_read = 1;
            break;

        case 's': //'s', 'S', and 'r' are mutually exclusive:
            if (!++io_pattern)
                ytc->ytc_io_pattern = YAPIO_IOP_SEQUENTIAL;
            break;

        case 'S':
            if (!++io_pattern)
                ytc->ytc_io_pattern = YAPIO_IOP_STRIDED;
            break;

        case 'R':
            if (!++io_pattern)
                ytc->ytc_io_pattern = YAPIO_IOP_RANDOM;
            break;

        case 'L':
            if (!++locality)
                ytc->ytc_remote_locality = 0;
            break;
        case 'D':
            if (!++locality)
                ytc->ytc_remote_locality = 1;
            break;
        case 'b':
            ytc->ytc_backwards = 1;
            break;
        case 'h':
            ytc->ytc_leave_holes = 1;
            break;
        case 'f':
            ytc->ytc_no_fsync = 1;
            break;

        case ',':
            ytg->ytg_num_contexts++;
            rw = locality = io_pattern = -1;
            break;

        default:
            log_msg_r0(YAPIO_LL_ERROR,
                       "invalid recipe character '%c': test %d, pos %u",
                       c, test_ctx_idx, i);
            return -EINVAL;
        }
    }

    if (rw)
    {
        if (rw > 0)
        {
            log_msg_r0(YAPIO_LL_ERROR,
                       "'r', 'w' are mutually exclusive: test %d, pos %u",
                       test_ctx_idx, i);
        }
        else
        {
            log_msg_r0(YAPIO_LL_ERROR, "'r' or 'w' not specified: test %d",
                       test_ctx_idx);
        }

        rc = -EINVAL;
    }
    else if (io_pattern)
    {
        if (io_pattern > 0)
        {
            log_msg_r0(YAPIO_LL_ERROR,
                       "'s', 'S', 'R' are mutually exclusive: test %d, pos %u",
                       test_ctx_idx, i);
        }
        else
        {
            log_msg_r0(YAPIO_LL_ERROR,
                       "'s', 'S' or 'R' not specified: test %d",
                       test_ctx_idx);
        }

        rc = -EINVAL;
    }
    else if (locality > 0)
    {
        if (locality > 0)
        {
            log_msg_r0(YAPIO_LL_ERROR,
                       "L', 'D' are mutually exclusive: test %d, pos %u",
                       test_ctx_idx, i);
        }
        rc = -EINVAL;
    }

    return rc;
}

static int
yapio_test_group_mpi_group_init(int start_rank, yapio_test_group_t *ytg)
{
    MPI_Group group_world;

    ytg->ytg_first_rank = start_rank;

    int rc = MPI_Comm_group(MPI_COMM_WORLD, &group_world);
    if (rc != MPI_SUCCESS)
    {
        log_msg(YAPIO_LL_ERROR, "MPI_Comm_group: %d", rc);
        return rc;
    }

    int *ranks = calloc(ytg->ytg_num_ranks, sizeof(int));
    if (!ranks)
        return -errno;

    int i;
    for (i = 0; i < ytg->ytg_num_ranks; i++)
        ranks[i] = start_rank + i;

    rc = MPI_Group_incl(group_world, ytg->ytg_num_ranks, ranks,
                        &ytg->ytg_group);

    free(ranks);

    if (rc != MPI_SUCCESS)
    {
        log_msg(YAPIO_LL_ERROR, "MPI_Group_incl: %d", rc);
        rc = -ECOMM;
    }
    else
    {
        rc = MPI_Comm_create(MPI_COMM_WORLD, ytg->ytg_group, &ytg->ytg_comm);
        if (rc != MPI_SUCCESS)
            log_msg(YAPIO_LL_ERROR, "MPI_Comm_create: %d", rc);
    }

    return rc;
}

static int
yapio_test_groups_setup_nranks(void)
{
    int i;
    int num_test_groups_without_nranks = 0;
    int num_ranks_counted = 0;

    for (i = 0; i < yapioNumTestGroups; i++)
    {
        yapio_test_group_t *ytg = &yapioTestGroups[i];

        if (!ytg->ytg_num_ranks)
            num_test_groups_without_nranks++;
        else
            num_ranks_counted += ytg->ytg_num_ranks;
    }

    /* If the user specified num_ranks for each test group then the tally
     * must match the num_ranks value reported by MPI.
     */
    if (!num_test_groups_without_nranks && num_ranks_counted != yapioNumRanks)
    {
        log_msg_r0(YAPIO_LL_ERROR,
                   "Test group rank tally (%d) does not match mpi nranks (%d)",
                   num_ranks_counted, yapioNumRanks);
        return -EINVAL;
    }

    bool slack_assigned = false;
    int num_ranks_recounted = 0;

    for (i = 0; i < yapioNumTestGroups; i++)
    {
        yapio_test_group_t *ytg = &yapioTestGroups[i];

        if (!ytg->ytg_num_ranks)
        {
            /* Divvy the remaining unassigned ranks amongst the test groups
             * which were not explicitly assigned rank counts.
             */
            ytg->ytg_num_ranks = ((yapioNumRanks - num_ranks_counted) /
                                  num_test_groups_without_nranks);
            if (!slack_assigned)
            {
                ytg->ytg_num_ranks += ((yapioNumRanks - num_ranks_counted) %
                                       num_test_groups_without_nranks);
                slack_assigned = true;
            }

            /* Ensure there are enough ranks to cover all test groups.
             */
            if (!ytg->ytg_num_ranks)
            {
                log_msg(YAPIO_LL_ERROR,
                        "No available ranks for test group %d", i);

                return -EINVAL;
            }
        }

        int rc = yapio_test_group_mpi_group_init(num_ranks_recounted, ytg);
        if (rc)
            return rc;

        num_ranks_recounted += ytg->ytg_num_ranks;
    }

    /* Sanity check.
     */
    if (num_ranks_recounted != yapioNumRanks)
        log_msg(YAPIO_LL_FATAL,
                "num_ranks_recounted (%d) != yapioNumRanks (%d)",
                num_ranks_recounted, yapioNumRanks);

    return 0;
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
        case 'D':
            yapioDecomposeCnt = MIN(atoi(optarg), YAPIO_DECOMPOSE_MAX);
            break;
        case 'd':
            yapioDbgLevel = MIN(atoi(optarg), YAPIO_LL_MAX);
            break;
        case 'F':
            yapioFilePerProcess = true;
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
        case 'P':
            yapioPolluteBlks = true;
            break;
        case 's':
            yapioDisplayStats = true;
            break;
        case 't':
            if (yapio_parse_test_recipe(optarg))
                yapio_print_help(YAPIO_EXIT_ERR);

            yapioNumTestGroups++;
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

//XXX this check needs to be moved
    if ((yapioNumBlksPerRank * yapioBlkSz) > YAPIO_MAX_SIZE_PER_PE)
        log_msg(YAPIO_LL_FATAL,
                "Per rank data size (%zu) exceeds max (%llu)",
                (yapioNumBlksPerRank * yapioBlkSz), YAPIO_MAX_SIZE_PER_PE);

    if (yapioDecomposeCnt && yapioNumRanks % (1 << yapioDecomposeCnt))
    {
        if (yapio_global_leader_rank())
        {
            log_msg(YAPIO_LL_FATAL,
                    "Decompose count must be greater than and a multiple of"
                    " mpi ranks. %d %d", yapioNumRanks, yapioDecomposeCnt);
        }
        else
        {
            yapio_exit(YAPIO_EXIT_ERR);
        }
    }

    int rc = yapio_test_groups_setup_nranks();
    if (rc)
        yapio_exit(rc);

    if (yapio_global_leader_rank())
    {
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

static int
yapio_make_fpp_filename(char *file_name, int rank)
{
    if (snprintf(file_name, PATH_MAX, "%s.%d",
                 yapioTestFileName, rank) > PATH_MAX)
    {
        log_msg(YAPIO_LL_ERROR, "%s", strerror(ENAMETOOLONG));
        return -ENAMETOOLONG;
    }

    return 0;
}

static int
yapio_open_fpp_file(int rank, int oflags)
{
    if (rank >= yapioNumRanks)
        return -EINVAL;

    char file_per_process_name[PATH_MAX + 1];

    int rc = yapio_make_fpp_filename(file_per_process_name, rank);
    if (rc)
        return rc;

    if (yapioFileDescFpp[rank] < 0)
    {
        yapioFileDescFpp[rank] = open(file_per_process_name, oflags, 0644);

        if (yapioFileDescFpp[rank] < 0)
        {
            rc = -errno;
            log_msg(YAPIO_LL_ERROR, "open `%s': %s",
                    file_per_process_name, strerror(errno));
        }
        else
        {
            log_msg(YAPIO_LL_DEBUG, "%s", file_per_process_name);
        }
    }

    return rc;
}

/**
 * yapio_setup_test_file - Rank0 will create a temp file and broadcast
 *    the name the to the other ranks who will then also open the temp file.
 */
static void
yapio_setup_test_file(const yapio_test_group_t *ytg)
{
    yapio_verify_test_directory();

    int path_len = snprintf(yapioTestFileName, PATH_MAX, "%s/%s%d_%s",
                            yapioTestRootDir, yapioFilePrefix,
                            ytg->ytg_group_num, YAPIO_MKSTEMP_TEMPLATE);
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
    MPI_OP_START;
    MPI_Bcast(&yapioTestFileName[path_len - YAPIO_MKSTEMP_TEMPLATE_LEN],
              YAPIO_MKSTEMP_TEMPLATE_LEN, MPI_CHAR, 0,
              yapioMyTestGroup->ytg_comm);
    MPI_OP_END;

    if (ytg->ytg_file_per_process)
    {
        if (yapio_leader_rank())
            unlink(yapioTestFileName);

        yapioFileDescFpp = malloc(yapioNumRanks * sizeof(int));
        if (!yapioFileDescFpp)
            log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));

        int i;
        for (i = 0; i < yapioNumRanks; i++)
            yapioFileDescFpp[i] = -1;

        int rc = yapio_open_fpp_file(yapio_relative_rank_get(ytg, 0),
                                     O_CREAT | O_EXCL | O_RDWR);
        if (rc)
            log_msg(YAPIO_LL_FATAL, "yapio_open_fpp_file: %s", strerror(-rc));

        yapio_make_fpp_filename(yapioTestFileNameFpp,
                                yapio_relative_rank_get(ytg, 0));

        log_msg(YAPIO_LL_DEBUG, "yapioTestFileNameFpp=%s",
                yapioTestFileNameFpp);
    }
    else
    {
        if (!yapio_leader_rank())
        {
            yapioFileDesc = open(yapioTestFileName, O_RDWR);
            if (yapioFileDesc < 0)
                log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));
        }
    }
}

static int
yapio_get_fd(int rank)
{
    return yapioMyTestGroup->ytg_file_per_process ?
        yapioFileDescFpp[rank] : yapioFileDesc;
}

static void
yapio_close_test_file(const yapio_test_group_t *ytg)
{
    int rc;

    if (ytg->ytg_file_per_process)
    {
        int i;
        for (i = 0; i < ytg->ytg_num_ranks; i++)
        {
            if (yapioFileDescFpp[i] >= 0)
            {
                rc = close(yapioFileDescFpp[i]);
                if (rc < 0)
                    break;
            }
        }
    }
    else
    {
        rc = close(yapioFileDesc);
    }

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
yapio_alloc_buffers(const yapio_test_group_t *ytg)
{
    yapioSourceBlkMd = calloc(ytg->ytg_num_blks_per_rank,
                              sizeof(yapio_blk_md_t));
    if (yapioSourceBlkMd == NULL)
        yapio_destroy_buffers_and_abort();

    yapioIOBuf = calloc(1, ytg->ytg_blk_sz);
    if (yapioIOBuf == NULL)
        yapio_destroy_buffers_and_abort();
}

static void
yapio_initialize_source_md_buffer(const yapio_test_group_t *ytg)
{
    int rel_rank = yapio_relative_rank_get(ytg, 0);
    size_t i;

    for (i = 0; i < ytg->ytg_num_blks_per_rank; i++)
    {
        yapioSourceBlkMd[i].ybm_writer_rank = yapioMyRank;

        yapioSourceBlkMd[i].ybm_blk_number = ytg->ytg_file_per_process ? i :
            rel_rank * ytg->ytg_num_blks_per_rank + i;

        yapioSourceBlkMd[i].ybm_owner_rank_fpp =
            ytg->ytg_file_per_process ? rel_rank : 0;
    }
}

static void
yapio_source_md_update_writer_rank(size_t source_md_idx, int new_writer_rank)
{
    if (source_md_idx >= yapioMyTestGroup->ytg_num_blks_per_rank)
        log_msg(YAPIO_LL_FATAL, "out of bounds source_md_idx=%zu",
                source_md_idx);

    yapioSourceBlkMd[source_md_idx].ybm_writer_rank = new_writer_rank;
}

static unsigned long long
yapio_get_content_word(const yapio_blk_md_t *md, size_t word_num)
{
    unsigned long long content_word =
        (yapio_get_blk_magic(md->ybm_blk_number) + md->ybm_writer_rank +
         md->ybm_blk_number + md->ybm_owner_rank_fpp + word_num);

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

yapio_blk_md_t *
yapio_test_ctx_to_md_array(const yapio_test_ctx_t *ytc,
                           enum yapio_test_ctx_mdh_in_out in_out, int *num_ops)
{
    const yapio_test_ctx_md_t *ytcmh = &ytc->ytc_in_out_md_ops[in_out];

    if (num_ops)
        *num_ops = ytcmh->ytcmh_num_ops;

    yapio_blk_md_t *md_array = ytcmh->ytcmh_ops;

    return md_array;
}

static int
yapio_fsync(void)
{
    int rc = 0;

    if (yapioMyTestGroup->ytg_file_per_process)
    {
        int i;
        for (i = 0; i < yapioNumRanks; i++)
        {
            if (yapioFileDescFpp[i] < 0)
                continue;

            if (fsync(yapioFileDescFpp[i]))
            {
                rc = errno;
                break;
            }
        }
    }
    else
    {
        if (fsync(yapioFileDesc))
            rc = errno;
    }

    if (rc)
    {
        log_msg(YAPIO_LL_ERROR, "fsync(): %s", strerror(rc));
        rc = -errno;
    }

    return rc;
}

/**
 * yapio_initialize_test_file_contents - each rank in the job writes its
 *    implicitly assigned set of contiguous blocks.
 */
static int
yapio_perform_io(yapio_test_ctx_t *ytc)
{
    yapio_test_group_t *ytg = ytc->ytc_group;
    int rc = 0;
    int num_ops_in = 0;

    const yapio_blk_md_t *md_array =
        yapio_test_ctx_to_md_array(ytc, YAPIO_TEST_CTX_MDH_IN, &num_ops_in);

    int j;
    for (j = 0; j < num_ops_in; j++)
    {
        const yapio_blk_md_t *md = &md_array[j];

        if (!ytc->ytc_read)
            yapio_apply_contents_to_io_buffer(yapioIOBuf, ytg->ytg_blk_sz, md);

        /* Obtain this IO's offset from the
         */
        off_t off = yapio_get_rw_offset(md, ytg->ytg_blk_sz);
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
            size_t adjusted_io_len = ytg->ytg_blk_sz - io_bytes;
            off_t adjusted_off = off + io_bytes;
            int fd_idx = md->ybm_owner_rank_fpp;
            int fd = yapio_get_fd(fd_idx);

            io_rc = ytc->ytc_read ?
                pread(fd, adjusted_buf, adjusted_io_len, adjusted_off) :
                pwrite(fd, adjusted_buf, adjusted_io_len, adjusted_off);

            log_msg(YAPIO_LL_DEBUG, "%s rc=%zd off=%lu@%d fr=%d",
                    ytc->ytc_read ? "pread" : "pwrite", io_rc, adjusted_off,
                    fd_idx, ytg->ytg_first_rank);

            if (io_rc > 0)
                io_bytes += io_rc;

        } while (io_rc > 0 && (size_t)io_bytes < ytg->ytg_blk_sz);

        if (io_rc < 0)
        {
            log_msg(YAPIO_LL_ERROR, "io failed at offset %lu: %s",
                    off, strerror(errno));
            rc = -errno;
            break;
        }

        if (ytc->ytc_read)
        {
            rc = yapio_verify_contents_of_io_buffer(yapioIOBuf,
                                                    ytg->ytg_blk_sz, md);
            if (rc)
                break;
        }
    }

    if (!rc && !ytc->ytc_no_fsync)
        rc = yapio_fsync();
    if (rc)
    {
        rc = -errno;
        log_msg(YAPIO_LL_ERROR, "fsync(): %s", strerror(errno));
    }

    return rc;
}

static void
yapio_setup_buffers(const yapio_test_group_t *ytg)
{
    yapio_alloc_buffers(ytg);
    yapio_initialize_source_md_buffer(ytg);
}

static void
yapio_unlink_test_file(void)
{
    bool fpp = yapioMyTestGroup->ytg_file_per_process;

    const char *unlink_fn = fpp ? yapioTestFileNameFpp : yapioTestFileName;
    log_msg(YAPIO_LL_DEBUG, "%s keep=%d", unlink_fn, yapioKeepFile);

    if (yapioKeepFile || (!fpp && !yapio_leader_rank()))
        return;

    int rc = unlink(unlink_fn);
    if (rc)
    {
        log_msg(YAPIO_LL_ERROR, "unlink %s: %s", unlink_fn, strerror(errno));

        yapio_exit(YAPIO_EXIT_ERR);
    }
}

static void
yapio_test_ctx_release_md(yapio_test_ctx_md_t *ytcmh)
{
    if (ytcmh->ytcmh_ops)
    {
        free(ytcmh->ytcmh_ops);
        ytcmh->ytcmh_ops = NULL;
    }
}

static void
yapio_test_ctx_release(yapio_test_ctx_t *ytc)
{
    int i;
    for (i = 0; i < YAPIO_TEST_CTX_MDH_MAX; i++)
        yapio_test_ctx_release_md(&ytc->ytc_in_out_md_ops[i]);
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

static yapio_blk_md_t *
yapio_test_context_alloc(yapio_test_ctx_t *ytc,
                         enum yapio_test_ctx_mdh_in_out in_out, size_t nblks)
{
    if (in_out >= YAPIO_TEST_CTX_MDH_MAX)
    {
        errno = EINVAL;
        return NULL;
    }

    yapio_test_ctx_md_t *ytcmh = &ytc->ytc_in_out_md_ops[in_out];

    ytcmh->ytcmh_num_ops = nblks;
    ytcmh->ytcmh_ops = calloc(nblks, sizeof(yapio_blk_md_t));

    return ytcmh->ytcmh_ops;
}

static void
yapio_test_context_sequential_setup_for_rank(yapio_test_ctx_t *ytc, int rank)
{
    int num_ops = 0;

    /* The local mode test case does not require the 'MDH_OUT' buffers at all.
     */
    enum yapio_test_ctx_mdh_in_out in_out = ytc->ytc_remote_locality ?
        YAPIO_TEST_CTX_MDH_OUT : YAPIO_TEST_CTX_MDH_IN;

    yapio_blk_md_t *md =
        yapio_test_ctx_to_md_array(ytc, in_out, &num_ops);

    if (!md)
        log_msg(YAPIO_LL_FATAL, "ytc_ops_md[%d] is NULL", rank);

    int i;
    for (i = 0; i < num_ops; i++)
    {
        size_t src_idx = ytc->ytc_backwards ? num_ops - i - 1 : i;

        if (!ytc->ytc_read)
            yapio_source_md_update_writer_rank(src_idx, rank);

        md[i] = yapioSourceBlkMd[src_idx];
        log_msg(YAPIO_LL_TRACE, "writer_rank=%d md->ybm_blk_number=%zd",
                md[i].ybm_writer_rank, md[i].ybm_blk_number);
    }
}

static int
yapio_blk_md_randomize(const yapio_blk_md_t *md_in, yapio_blk_md_t *md_out,
                       size_t num_mds, bool initialize_md_out)
{
    size_t buf_sz = num_mds * sizeof(int);
    int *array_of_randoms = malloc(buf_sz);
    if (!array_of_randoms)
        return -ENOMEM;

    int rc = yapio_read_from_dev_urandom((void *)array_of_randoms, buf_sz);
    if (rc)
        return rc;

    size_t i;
    if (initialize_md_out)
    {
        /* Initialize the output array with a copy of the input array.
         */
        for (i = 0; i < num_mds; i++)
            md_out[i] = md_in[i];
    }

    /* Swap output array items based on the random data in the array.
     */
    for (i = 0; i < num_mds; i++)
    {
        size_t swap_idx = array_of_randoms[i] % num_mds;
        yapio_blk_md_t md_tmp = md_out[swap_idx];
        md_out[swap_idx] = md_out[i];
        md_out[i] = md_tmp;

        log_msg(YAPIO_LL_TRACE, "swapped %zu:%zu <-> %zu:%zu",
                i, md_out[swap_idx].ybm_blk_number, swap_idx,
                md_out[i].ybm_blk_number);
    }

    free(array_of_randoms);

    return 0;
}

/**
 * yapio_test_context_setup_local - setup driver for the 'local' I/O
 *    mode where each rank determines its own work queue.
 * Note: YAPIO_IOP_STRIDED mode is not supported in this mode.
 */
static int
yapio_test_context_setup_local(yapio_test_ctx_t *ytc)
{
    if (ytc->ytc_remote_locality)
    {
        return -EINVAL;
    }
    else if (ytc->ytc_io_pattern == YAPIO_IOP_STRIDED)
    {
        return -ENOTSUP;
    }
    else if (ytc->ytc_io_pattern != YAPIO_IOP_SEQUENTIAL &&
             ytc->ytc_io_pattern != YAPIO_IOP_RANDOM)
    {

        log_msg(YAPIO_LL_WARN, "unknown io_pattern=%d", ytc->ytc_io_pattern);
        return -EBADRQC;
    }

    yapio_test_group_t *ytg = ytc->ytc_group;

    /* In this mode all operation instructions are pulled from the local
     * Source Blk Metadata - there is no exchange with other ranks.
     */
    yapio_blk_md_t *md =
        yapio_test_context_alloc(ytc, YAPIO_TEST_CTX_MDH_IN,
                                 ytg->ytg_num_blks_per_rank);
    if (!md)
        return -errno;

    int rc = 0;

    if (ytc->ytc_io_pattern == YAPIO_IOP_SEQUENTIAL)
        yapio_test_context_sequential_setup_for_rank(ytc, yapioMyRank -
                                                     ytg->ytg_first_rank);

    else if (ytc->ytc_io_pattern == YAPIO_IOP_RANDOM)
        rc = yapio_blk_md_randomize(yapioSourceBlkMd, md,
                                    ytg->ytg_num_blks_per_rank, true);

    return rc;
}

/**
 * yapio_test_context_setup_distributed_sequential - Function which handles the
 *    exchange of blk metadata on behalf of a distributed, sequential
 *    operation.  The distribution method is to send the entire blk metadata
 *    set to 'my_rank + 1' and receive from 'my_rank -1'.
 */
static int
yapio_test_context_setup_distributed_sequential(yapio_test_ctx_t *ytc)
{
    if (ytc->ytc_io_pattern != YAPIO_IOP_SEQUENTIAL)
        return -EINVAL;

    const yapio_test_group_t *ytg = ytc->ytc_group;
    const size_t nblks_per_rank = ytg->ytg_num_blks_per_rank;

    int recv_rank = yapio_relative_rank_get(ytg, -1);
    int dest_rank = yapio_relative_rank_get(ytg, 1);

    yapio_blk_md_t *md_send =
        yapio_test_context_alloc(ytc, YAPIO_TEST_CTX_MDH_OUT, nblks_per_rank);

    yapio_blk_md_t *md_recv =
        yapio_test_context_alloc(ytc, YAPIO_TEST_CTX_MDH_IN, nblks_per_rank);

    yapio_test_context_sequential_setup_for_rank(ytc, dest_rank);

    int send_recv_cnt = sizeof(yapio_blk_md_t) * nblks_per_rank;
    int send_recv_tag = YAPIO_IOP_SEQUENTIAL;
    MPI_Status status; //unused

    log_msg(YAPIO_LL_TRACE, "dest_rank=%d recv_rank=%d", dest_rank, recv_rank);

    MPI_OP_START;
    int rc = MPI_Sendrecv((void *)md_send, send_recv_cnt, MPI_BYTE, dest_rank,
                          send_recv_tag, (void *)md_recv, send_recv_cnt,
                          MPI_BYTE, recv_rank, send_recv_tag,
                          yapioMyTestGroup->ytg_comm, &status);
    MPI_OP_END;

    if (rc != MPI_SUCCESS)
        log_msg(YAPIO_LL_ERROR, "MPI_Sendrecv: %d", rc);

    return rc;
}

static int
yapio_test_context_setup_distributed_random_or_strided(yapio_test_ctx_t *ytc)
{
    const yapio_test_group_t *ytg = ytc->ytc_group;
    const size_t nblks_per_rank = ytg->ytg_num_blks_per_rank;
    const int nranks = ytg->ytg_num_ranks;

    if (ytc->ytc_io_pattern != YAPIO_IOP_RANDOM &&
        ytc->ytc_io_pattern != YAPIO_IOP_STRIDED)
        return -EINVAL;

    else if (nblks_per_rank % nranks)
        return -EINVAL;

    yapio_blk_md_t *md_recv =
        yapio_test_context_alloc(ytc, YAPIO_TEST_CTX_MDH_IN, nblks_per_rank);

    if (!md_recv)
        return -errno;

    yapio_blk_md_t *md_send =
        yapio_test_context_alloc(ytc, YAPIO_TEST_CTX_MDH_OUT, nblks_per_rank);

    if (!md_send)
        return -errno;

    const bool strided =
        ytc->ytc_io_pattern == YAPIO_IOP_STRIDED ? true : false;

    size_t src_idx;
    const int nblks_div_nranks = nblks_per_rank / nranks;

//XXX open all FDs here for FPP
    if (strided)
    {
        int i, j, total = 0;
        for (i = 0, total = 0; i < nranks; i++)
            for (j = 0; j < nblks_div_nranks; j++, total++)
            {
                src_idx = (nranks * j) + i;
#if 0 //backwards + strided is broken
                if (ytc->ytc_backwards)
                    src_idx = (nranks * (nblks_div_nranks - j) - 1 + i);
#endif

                if (!ytc->ytc_read)
                    yapio_source_md_update_writer_rank(src_idx, i);

                md_send[total] = yapioSourceBlkMd[src_idx];
            }
    }
    else
    {
        yapio_blk_md_randomize(yapioSourceBlkMd, md_send, nblks_per_rank,
                               true);

        if (!ytc->ytc_read)
        {
            for (src_idx = 0; src_idx < nblks_per_rank; src_idx++)
            {
                int rank = src_idx / nblks_div_nranks;

                yapio_source_md_update_writer_rank(src_idx, rank);
                md_send[src_idx].ybm_writer_rank = rank;
            }
        }
    }

    int send_recv_cnt = nblks_div_nranks * sizeof(yapio_blk_md_t);

    MPI_OP_START;
    int rc = MPI_Alltoall(md_send, send_recv_cnt, MPI_BYTE,
                          md_recv, send_recv_cnt, MPI_BYTE,
                          yapioMyTestGroup->ytg_comm);
    MPI_OP_END;

    if (rc != MPI_SUCCESS)
    {
        log_msg(YAPIO_LL_ERROR, "MPI_Alltoall: %d", rc);
        return rc;
    }

    return strided ? 0 : yapio_blk_md_randomize(md_recv, md_recv,
                                                nblks_per_rank, false);
}

static int
yapio_test_context_setup_distributed(yapio_test_ctx_t *ytc)
{
    enum yapio_patterns pattern = ytc->ytc_io_pattern;

    if (!ytc->ytc_remote_locality)
    {
        return -EINVAL;
    }
    else if (pattern != YAPIO_IOP_SEQUENTIAL &&
             pattern != YAPIO_IOP_RANDOM &&
             pattern != YAPIO_IOP_STRIDED)
    {
        log_msg(YAPIO_LL_WARN, "unknown io_pattern=%d", pattern);
        return -EBADRQC;
    }

    return pattern == YAPIO_IOP_SEQUENTIAL ?
        yapio_test_context_setup_distributed_sequential(ytc) :
        yapio_test_context_setup_distributed_random_or_strided(ytc);
}

static int
yapio_prepare_fpp_file_desc(const yapio_test_ctx_t *ytc)
{
    int rc = 0;

    if (ytc->ytc_group->ytg_file_per_process)
    {
        int num_ops = 0;

        const yapio_blk_md_t *md_in =
            yapio_test_ctx_to_md_array(ytc, YAPIO_TEST_CTX_MDH_IN, &num_ops);

        int i = 0;
        for (i = 0; i < num_ops; i++)
        {
            rc = yapio_open_fpp_file(md_in[i].ybm_owner_rank_fpp, O_RDWR);
            if (rc)
                break;
        }
    }

   return rc;
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
yapio_test_context_setup(yapio_test_ctx_t *ytc, const int test_num)
{
    ytc->ytc_test_num = test_num;
    ytc->ytc_run_status = YAPIO_TEST_CTX_RUN_NOT_STARTED;

    int rc = ytc->ytc_remote_locality ?
        yapio_test_context_setup_distributed(ytc) :
        yapio_test_context_setup_local(ytc);

    if (rc)
    {
        log_msg(YAPIO_LL_ERROR,
                "yapio_test_context_setup_%s: %s",
                ytc->ytc_remote_locality ? "distributed" : "local",
                strerror(-rc));

        yapio_test_ctx_release(ytc);

        return rc;
    }

    return yapio_prepare_fpp_file_desc(ytc);
}

static void
yapio_verify_test_contexts(yapio_test_group_t *ytg)
{
    if (!ytg->ytg_num_contexts)
    {
        log_msg(YAPIO_LL_DEBUG, "Using default tests");
        /* Perform the default test - local write followed by local read.
         */
        ytg->ytg_num_contexts = 2;
        yapio_test_ctx_t *ytc_write = &ytg->ytg_contexts[0];
        yapio_test_ctx_t *ytc_read  = &ytg->ytg_contexts[1];;

        ytc_write->ytc_io_pattern = ytc_read->ytc_io_pattern =
            YAPIO_IOP_SEQUENTIAL;

        ytc_read->ytc_read = 1;
    }

    int i;
    for (i = 0; i < ytg->ytg_num_contexts; i++)
    {
        yapio_test_ctx_t *ytc = &ytg->ytg_contexts[i];

        if ((ytc->ytc_io_pattern == YAPIO_IOP_RANDOM ||
             ytc->ytc_io_pattern == YAPIO_IOP_STRIDED) &&
            ytg->ytg_num_blks_per_rank % ytg->ytg_num_ranks)
        {
            if (yapio_leader_rank())
            {
                log_msg(YAPIO_LL_FATAL,
                        "random and strided tests require BlksPerRank (%zu) "
                        "to be a multiple of NumRanks (%d)",
                        ytg->ytg_num_blks_per_rank, ytg->ytg_num_ranks);
            }
            else
            {
                yapio_exit(-EINVAL);
            }
        }
    }
}

static float
yapio_timer_to_float(const yapio_timer_t *timer)
{
    float bwait = timer->tv_sec + (float)((float)timer->tv_nsec /
                                          YAPIO_NSEC_PER_SEC);

    return bwait;
}

static int
yapio_gather_barrier_stats_median_cmp(const void *a, const void *b)
{
    float a_val = yapio_timer_to_float((const yapio_timer_t *)a);
    float b_val = yapio_timer_to_float((const yapio_timer_t *)b);

    if (a_val > b_val)
        return 1;
    else if (a_val < b_val)
        return -1;

    return 0;
}

static void
yapio_gather_barrier_stats(const yapio_timer_t *barrier_timer_this_rank,
                           float *barrier_global_results,
                           int *barrier_max_rank)
{
    yapio_timer_t *all_barrier_timers = NULL;

    if (barrier_global_results)
    {
        *barrier_max_rank = -1;
        barrier_global_results[YAPIO_BARRIER_STATS_MAX] = 0.0;
        barrier_global_results[YAPIO_BARRIER_STATS_AVG] = 0.0;

        /* This node is rank0 and will gather the timers for reporting.
         */
        all_barrier_timers = calloc(yapioNumRanks, sizeof(yapio_timer_t));

        if (!all_barrier_timers)
            log_msg(YAPIO_LL_FATAL, "calloc: %s", strerror(ENOMEM));
    }

    MPI_OP_START;
    int rc = MPI_Gather(barrier_timer_this_rank, sizeof(yapio_timer_t),
                        MPI_BYTE, all_barrier_timers, sizeof(yapio_timer_t),
                        MPI_BYTE, 0, yapioMyTestGroup->ytg_comm);
    MPI_OP_END;

    if (rc != MPI_SUCCESS)
        log_msg(YAPIO_LL_FATAL, "MPI_Gather: error=%d", rc);

    if (barrier_global_results)
    {
        int i;
        for (i = 0; i < yapioNumRanks; i++)
        {
            float bwait = yapio_timer_to_float(&all_barrier_timers[i]);

            barrier_global_results[YAPIO_BARRIER_STATS_AVG] += bwait;

            if (bwait > barrier_global_results[YAPIO_BARRIER_STATS_MAX])
            {
                barrier_global_results[YAPIO_BARRIER_STATS_MAX] = bwait;
                *barrier_max_rank = i;
            }
        }
        /* Calculate average
         */
        barrier_global_results[YAPIO_BARRIER_STATS_AVG] /= yapioNumRanks;

        /* Determine the median by sorting the result array.
         */
        qsort((void *)all_barrier_timers, (size_t)yapioNumRanks,
              sizeof(yapio_timer_t), yapio_gather_barrier_stats_median_cmp);

        barrier_global_results[YAPIO_BARRIER_STATS_MED] =
            yapio_timer_to_float(&all_barrier_timers[yapioNumRanks / 2]);

        free(all_barrier_timers);
    }
}

static void
yapio_gather_test_barrier_results(yapio_test_ctx_t *ytc)
{
    if (yapioDisplayStats)
    {
        yapio_leader_rank() ?
            yapio_gather_barrier_stats(ytc->ytc_barrier_wait,
                                       ytc->ytc_barrier_results,
                                       &ytc->ytc_barrier_max_rank) :
            yapio_gather_barrier_stats(ytc->ytc_barrier_wait, NULL, NULL);
    }
}

static void
yapio_stat_ready(void)
{
    pthread_mutex_lock(&yapioThreadMutex);
    pthread_cond_signal(&yapioThreadCond);
    pthread_mutex_unlock(&yapioThreadMutex);
}

static void
yapio_send_test_results(yapio_test_ctx_t *ytc)
{
    if (!yapio_leader_rank())
        return;

    if (ytc->ytc_group->ytg_group_num > 0)
    {
        //XXx sending the whole ctx is not a good idea
        MPI_OP_START;
        int rc = MPI_Send(ytc, sizeof(yapio_test_ctx_t), MPI_BYTE, 0,
                          ytc->ytc_test_num, MPI_COMM_WORLD);
        MPI_OP_END;

        if (rc != MPI_SUCCESS)
            log_msg(YAPIO_LL_FATAL, "MPI_Send failed: %d", rc);
    }
    else
    {
        /* Local operation to wake up stats reporting thread.
         */
        yapio_stat_ready();
    }

    log_msg(YAPIO_LL_DEBUG, "ytc_test_num=%d", ytc->ytc_test_num);
}

static void
yapio_display_result(const yapio_test_ctx_t *ytc)
{
    const yapio_timer_t *test_duration = &ytc->ytc_test_duration;
    const int nranks = ytc->ytc_group->ytg_num_ranks;
    const size_t nblks_per_rank = ytc->ytc_group->ytg_num_blks_per_rank;
    const size_t blksz = ytc->ytc_group->ytg_blk_sz;

    float bandwidth =
        (float)(((float)nranks * nblks_per_rank * blksz) /
                yapio_timer_to_float(test_duration));

    char *unit_str;
    if (bandwidth < (1ULL << 20))
    {
        bandwidth /= (1ULL << 10);
        unit_str = "K";
    }
    else if (bandwidth < (1ULL << 30))
    {
        bandwidth /= (1ULL << 20);
        unit_str = "M";
    }
    else if (bandwidth < (1ULL << 40))
    {
        bandwidth /= (1ULL << 30);
        unit_str = "G";
    }
    else if (bandwidth < (1ULL << 50))
    {
        bandwidth /= (1ULL << 40);
        unit_str = "T";
    }

    fprintf(stdout, "%8.2f %02d.%02d: %s%s%s%s%s%s %6.02f %siB/s%s",
            yapio_timer_to_float(&ytc->ytc_reported_time),
            ytc->ytc_group->ytg_group_num, ytc->ytc_test_num,
            (ytc->ytc_io_pattern == YAPIO_IOP_SEQUENTIAL ? "s" :
             (ytc->ytc_io_pattern ==
              YAPIO_IOP_RANDOM ? "R" : "S")),
            ytc->ytc_read            ? "r" : "w",
            ytc->ytc_remote_locality ? "D" : "L",
            ytc->ytc_no_fsync        ? "f" : "-",
            ytc->ytc_backwards       ? "b" : "-",
            ytc->ytc_leave_holes     ? "h" : "-",
            bandwidth, unit_str,
            yapioDisplayStats ? "" : "\n");

    if (yapioDisplayStats)
    {
        const float *barrier_results = ytc->ytc_barrier_results;
        int   barrier_max_rank = ytc->ytc_barrier_max_rank;

        fprintf(stdout,
                "  %8.2f <"YAPIO_TIME_PRINT_SPEC","YAPIO_TIME_PRINT_SPEC","
                YAPIO_TIME_PRINT_SPEC","YAPIO_TIME_PRINT_SPEC":%d>\n",
                YAPIO_TIMER_ARGS(test_duration),
                YAPIO_TIMER_ARGS(&ytc->ytc_setup_time),
                barrier_results[YAPIO_BARRIER_STATS_AVG],
                barrier_results[YAPIO_BARRIER_STATS_MED],
                barrier_results[YAPIO_BARRIER_STATS_MAX],
                barrier_max_rank);
    }
}

static void
yapio_exec_all_tests(void)
{
    yapio_test_group_t *ytg = yapioMyTestGroup;

    yapio_mpi_barrier(yapioMyTestGroup->ytg_comm);

    log_msg_r0(YAPIO_LL_DEBUG,
               "g@%p nctxs=%d nranks=%d nblks=%zu blksz=%zu fpp=%d lr=%d",
               ytg, ytg->ytg_num_contexts, ytg->ytg_num_ranks,
               ytg->ytg_num_blks_per_rank, ytg->ytg_blk_sz,
               ytg->ytg_file_per_process, ytg->ytg_leader_rank);

    int i;
    for (i = 0; i < yapioMyTestGroup->ytg_num_contexts; i++)
    {
        yapio_test_ctx_t *ytc = &yapioMyTestGroup->ytg_contexts[i];
        ytc->ytc_group = yapioMyTestGroup;
        /* Setup may require a notable amount of time for buffer exchange and
         * file descriptor operations.
         */
        if (yapio_leader_rank())
            yapio_start_timer(&ytc->ytc_setup_time);

        int rc = yapio_test_context_setup(ytc, i);
        if (rc)
            yapio_exit(rc);

        yapio_mpi_barrier(yapioMyTestGroup->ytg_comm); //setup barrier

        if (yapio_leader_rank())
            yapio_end_timer(&ytc->ytc_setup_time);

        ytc->ytc_run_status = YAPIO_TEST_CTX_RUN_STARTED;

        /* Sync all ranks before and after starting the test.
         */
        yapio_mpi_barrier(yapioMyTestGroup->ytg_comm);
        if (yapio_leader_rank())
            yapio_start_timer(&ytc->ytc_test_duration);

        yapio_perform_io(ytc);

        yapio_start_timer(&ytc->ytc_barrier_wait[0]);
        yapio_mpi_barrier(yapioMyTestGroup->ytg_comm);
        yapio_start_timer(&ytc->ytc_barrier_wait[1]);

        /* Result is in barrier[0]
         */
        yapio_get_time_duration(&ytc->ytc_barrier_wait[0],
                                &ytc->ytc_barrier_wait[1]);
        yapio_get_time_duration(&ytc->ytc_test_duration,
                                &ytc->ytc_barrier_wait[1]);

        ytc->ytc_run_status = YAPIO_TEST_CTX_RUN_COMPLETE;

        yapio_gather_test_barrier_results(ytc);
        yapio_send_test_results(ytc);

        /* Free memory allocated in the test.
         */
        yapio_test_ctx_release(ytc);
    }
}

static void
yapio_assign_rank_to_group(void)
{
    int total_ranks;
    int i;
    for (i = 0, total_ranks = 0; i < yapioNumTestGroups; i++)
    {
        yapio_test_group_t *ytg = &yapioTestGroups[i];

        if (yapioMyRank < total_ranks + ytg->ytg_num_ranks)
        {
            yapioMyTestGroup = ytg;
            if (yapioMyRank == total_ranks)
                ytg->ytg_leader_rank = true;

            break;
        }

        total_ranks += ytg->ytg_num_ranks;
    }

    if (!yapioMyTestGroup)
        log_msg(YAPIO_LL_FATAL, "rank=%d was not assigned to test group",
                yapioMyRank);
}

static void *
yapio_stats_reporting(void *unused_arg)
{
    int remaining_test_contexts_to_report;

    yapio_timer_t start_time;
    yapio_get_time(&start_time);

    do
    {
        remaining_test_contexts_to_report = 0;

        pthread_mutex_lock(&yapioThreadMutex);

        int i;
        for (i = 0; i < yapioNumTestGroups; i++)
        {
            yapio_test_group_t *ytg = &yapioTestGroups[i];
            int j;
            for (j = 0; j < ytg->ytg_num_contexts; j++)
            {
                yapio_test_ctx_t *ytc = &ytg->ytg_contexts[j];

                if (ytc->ytc_run_status == YAPIO_TEST_CTX_RUN_COMPLETE)
                {
                    ytc->ytc_reported_time = start_time;
                    yapio_end_timer(&ytc->ytc_reported_time);

                    ytc->ytc_run_status =
                        YAPIO_TEST_CTX_RUN_STATS_REPORTED;

                    yapio_display_result(ytc);
                }

                if (ytc->ytc_run_status != YAPIO_TEST_CTX_RUN_STATS_REPORTED)
                    remaining_test_contexts_to_report++;
            }
        }

        if (remaining_test_contexts_to_report)
            pthread_cond_wait(&yapioThreadCond, &yapioThreadMutex);

        pthread_mutex_unlock(&yapioThreadMutex);
    } while (remaining_test_contexts_to_report);

    return unused_arg;
}

static void *
yapio_stats_collection(void *unused_arg)
{
    ssize_t num_remote_test_contexts;
    int i;
    /* Skip the first group since the leader rank is 'this' rank.
     */
    for (i = 1, num_remote_test_contexts = 0; i < yapioNumTestGroups; i++)
        num_remote_test_contexts += yapioTestGroups[i].ytg_num_contexts;

    if (!num_remote_test_contexts)
        return NULL;

    log_msg(YAPIO_LL_DEBUG, "num_remote_test_contexts=%zd",
            num_remote_test_contexts);

    MPI_Request requests[num_remote_test_contexts];

    int req_num;
    for (i = 1, req_num = 0; i < yapioNumTestGroups; i++)
    {
        yapio_test_group_t *ytg = &yapioTestGroups[i];
        int j;
        for (j = 0; j < ytg->ytg_num_contexts; j++)
        {
            MPI_OP_START;
            int rc = MPI_Irecv(&ytg->ytg_contexts[j], sizeof(yapio_test_ctx_t),
                               MPI_BYTE, ytg->ytg_first_rank, j,
                               MPI_COMM_WORLD, &requests[req_num++]);
            MPI_OP_END;

            if (rc != MPI_SUCCESS)
                log_msg(YAPIO_LL_FATAL, "MPI_Irecv failed: %d", rc);
        }
    }

    int remaining_reports = num_remote_test_contexts;
    while (remaining_reports)
    {
        int index, flag;
        MPI_Status status;

        MPI_OP_START;
        int rc = MPI_Testany(num_remote_test_contexts, requests, &index, &flag,
                             &status);
        MPI_OP_END;

        if (rc != MPI_SUCCESS)
            log_msg(YAPIO_LL_FATAL, "MPI_Testany failed: %d", rc);

        if (index != MPI_UNDEFINED)
        {
            remaining_reports--;

            log_msg(YAPIO_LL_DEBUG, "index=%d rc=%d flag=%d rem=%d",
                    i, rc, flag, remaining_reports);

            yapio_stat_ready();
        }

        usleep(YAPIO_STATS_SLEEP_USEC);
    }

    return unused_arg;
}

static void
yapio_start_stats_collection_and_reporting_threads(void)
{
    if (pthread_create(&yapioStatsReportingThread, NULL,
                       yapio_stats_reporting, NULL))
        log_msg(YAPIO_LL_FATAL, "pthread_create: %s", strerror(errno));

    if (yapioNumTestGroups > 1)
    {
        /* This thread only starts if there are multiple test groups.
         */
        if (pthread_create(&yapioStatsCollectionThread, NULL,
                           yapio_stats_collection, NULL))
            log_msg(YAPIO_LL_FATAL, "pthread_create: %s", strerror(errno));
    }
}

static void
yapio_destroy_collection_and_reporting_threads(void)
{
    pthread_join(yapioStatsReportingThread, NULL);

    if (yapioNumTestGroups > 1)
        pthread_join(yapioStatsCollectionThread, NULL);
}

int
main(int argc, char *argv[])
{
    yapio_mpi_setup(argc, argv);

    yapio_getopts(argc, argv);

    yapio_assign_rank_to_group();

    yapio_verify_test_contexts(yapioMyTestGroup);

    yapio_setup_buffers(yapioMyTestGroup);

    yapio_setup_test_file(yapioMyTestGroup);

    if (yapio_global_leader_rank())
        yapio_start_stats_collection_and_reporting_threads();

    yapio_mpi_barrier(MPI_COMM_WORLD);

    yapio_exec_all_tests();

    yapio_close_test_file(yapioMyTestGroup);

    if (yapio_global_leader_rank())
        yapio_destroy_collection_and_reporting_threads();

    yapio_destroy_buffers();

    yapio_unlink_test_file();

    yapio_exit(YAPIO_EXIT_OK);

    return 0;
}
