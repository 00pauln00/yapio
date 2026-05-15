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
#include <sys/mman.h>
#include <sys/queue.h>
#ifdef YAPIO_IME
#include <im_client_native2.h>
#endif

#define YAPIO_CALLOC(n, size)                                           \
    ({                                                                  \
        void *buf = calloc(n, size);                                    \
        log_msg(YAPIO_LL_DEBUG, "alloc: %p %zu", buf, (n * size));      \
        buf;                                                            \
    })

#define YAPIO_MALLOC(size) YAPIO_CALLOC(1, size);

#define YAPIO_FREE(ptr)                             \
    {                                               \
        free((ptr));                                \
        log_msg(YAPIO_LL_DEBUG, "free: %p", (ptr)); \
    }

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define YAPIO_OPTS "b:N:hd:i:m:t:nPD:sV:S:v:C:z:Wq:"

#define YAPIO_DEF_NBLKS_PER_PE     1000
#define YAPIO_DEF_BLK_SIZE         4096
#define YAPIO_MAX_BLK_SIZE         (1ULL << 30)
#define YAPIO_MAX_SIZE_PER_PE      (1ULL << 40)
#define YAPIO_DEFAULT_FILE_PREFIX  "yapio."
#define YAPIO_MKSTEMP_TEMPLATE     "XXXXXX"
#define YAPIO_MKSTEMP_TEMPLATE_LEN 6
#define YAPIO_DECOMPOSE_MAX        8
#define YAPIO_RECIPE_PARAM_STRLEN_MAX 65

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
static int         yapioDbgLevel       = YAPIO_LL_WARN;
static bool        yapioMpiInit        = false;
static bool        yapioPolluteBlks    = false;
static bool        yapioNetOnly        = false;
static bool        yapioDisplayStats   = false;
static int         yapioDecomposeCnt   = 0;
static const char *yapioExecName;
static const char *yapioTestRootDir;
static char        yapioTestFileName[PATH_MAX + 1];
static char        yapioTestFileNameFpp[PATH_MAX + 1];
static char        yapioRestartFileName[PATH_MAX + 1];
static int         yapioMyRank;
static int         yapioNumRanks;
static int         yapioFileDesc;
static int        *yapioFileDescFpp;
static char       *yapioIOBuf;
static bool        yapioVerifyRead     = true;
static bool        yapioWordLevelVerify = false;
static const char *yapioTestFileNamePrefix = "";
static int         yapioStoneWallNsecs;
static bool        yapioStoneWalled = false;
static bool        yapioUseStoneWalling = false;
static bool        yapioHaltStonewallThread = false;
static int         yapioIOErrorCnt;

static pthread_t       yapioStatsCollectionThread;
static pthread_t       yapioStatsReportingThread;
static pthread_t       yapioStoneWallThread;
static pthread_cond_t  yapioThreadCond  = PTHREAD_COND_INITIALIZER;
static pthread_mutex_t yapioThreadMutex = PTHREAD_MUTEX_INITIALIZER;

#define YAPIO_STATS_SLEEP_USEC 10000

#define MPI_OP_START pthread_mutex_lock(&yapioThreadMutex)
#define MPI_OP_END   pthread_mutex_unlock(&yapioThreadMutex)

enum yapio_io_modes
{
    YAPIO_IO_MODE_DEFAULT = 0,
    YAPIO_IO_MODE_POSIX   = YAPIO_IO_MODE_DEFAULT,
    YAPIO_IO_MODE_IME     = 1,
    YAPIO_IO_MODE_MPIIO   = 2, // Not yet supported
    YAPIO_IO_MODE_MMAP    = 3,
    YAPIO_IO_MODE_NIOVA   = 4, // Direct niova block client (bypasses ublk)
    YAPIO_IO_MODE_LAST    = 5,
};

static enum yapio_io_modes yapioModeCurrent = YAPIO_IO_MODE_DEFAULT;

#ifdef YAPIO_NIOVA
#include <niova/nclient.h>
#include <niova/nclient_private.h>
#include <niova/common.h>

#include <niova/niova_block_common.h>

/* Required by niova log.h static-inline helpers (regFileEntry, logEntryFileInfo). */
REGISTRY_ENTRY_FILE_GENERATE;

#define YAPIO_NIOVA_BLOCK_SIZE 4096
/* 64 GiB — mirrors NIOVA_DEFAULT_FILE_SIZE from niova_block_common.h */
#define YAPIO_NIOVA_DEFAULT_FILE_SIZE ((size_t)1 << 36)

static char  *yapioNiovaVdevFile        = NULL;
static char   yapioNiovaConnectStr[256] = {0};

/* Per-rank niova state - each MPI rank (process) holds its own client
 * connected to the vdev assigned to it by line number in the vdev file.
 */
static niova_block_client_t *yapioNiovaClient  = NULL;
static uuid_t                yapioNiovaVdevUuid;

/* Total vdev size in bytes.  When non-zero, random IO picks start_vblk from
 * [0, yapioNiovaVdevSizeBytes/4096) — matching niova-block-test behaviour
 * where -N (nops) and the address range are independent.
 * Set via -z <bytes>.  0 means fall back to n*blk_sz (legacy behaviour).
 */
static size_t yapioNiovaVdevSizeBytes = 0;

/* Runtime max transfer size in vblks from niova_block_client_max_xfer_vblks(),
 * set after client connect.  Used to cap nblks per NiovaBlockClientReadv/Writev
 * call, mirroring niova-block-test's NIOVA_MAX_VBLKS_PER_REQUEST enforcement.
 */
static int yapioNiovaMaxXferVblks = 0;
#endif /* YAPIO_NIOVA */

typedef int     (*openf)(const char *, int, mode_t);
typedef int     (*closef)(int);
typedef int     (*fsyncf)(int);
typedef int     (*unlinkf)(const char *);
typedef ssize_t (*preadf)(int, void *, size_t, off_t);
typedef ssize_t (*pwritef)(int, const void *, size_t, off_t);

typedef struct yapio_io_syscall_ops
{
    openf   yapio_sys_open;
    closef  yapio_sys_close;
    fsyncf  yapio_sys_fsync;
    unlinkf yapio_sys_unlink;
    preadf  yapio_sys_pread;
    pwritef yapio_sys_pwrite;
} yapio_io_syscall_ops_t;

struct yapio_fd_map
{
    int   yfm_fd;
    void *yfm_addr;
};

#define YAPIO_FD_MAP_SIZE_MAX 131072

static struct yapio_fd_map yapioFdMap[YAPIO_FD_MAP_SIZE_MAX];

static const yapio_io_syscall_ops_t *yapio_io_modes[YAPIO_IO_MODE_LAST];

static const yapio_io_syscall_ops_t yapioDefaultSysCallOps =
    {.yapio_sys_open   = (openf)open,
     .yapio_sys_close  = close,
     .yapio_sys_fsync  = fsync,
     .yapio_sys_unlink = unlink,
     .yapio_sys_pread  = pread,
     .yapio_sys_pwrite = pwrite};

static int
yapio_mmap_open(const char *file, int flags, mode_t mode);

static int
yapio_mmap_close(int fd);

static int
yapio_mmap_fsync(int fd);

static ssize_t
yapio_mmap_pread(int fd, void *buf, size_t count, off_t offset);

static ssize_t
yapio_mmap_pwrite(int fd, const void *buf, size_t count, off_t offset);

static const yapio_io_syscall_ops_t yapioMmapSysCallOps =
    {.yapio_sys_open   = (openf)yapio_mmap_open,
     .yapio_sys_close  = yapio_mmap_close,
     .yapio_sys_fsync  = yapio_mmap_fsync,
     .yapio_sys_unlink = unlink,
     .yapio_sys_pread  = yapio_mmap_pread,
     .yapio_sys_pwrite = yapio_mmap_pwrite};

const yapio_io_syscall_ops_t *yapioSysCallOps = &yapioDefaultSysCallOps;

#ifdef YAPIO_IME
const yapio_io_syscall_ops_t yapioIMESysCallOps =
    {.yapio_sys_open   = ime_client_native2_open,
     .yapio_sys_close  = ime_client_native2_close,
     .yapio_sys_fsync  = ime_client_native2_fsync,
     .yapio_sys_unlink = ime_client_native2_unlink,
     .yapio_sys_pread  = (preadf)ime_client_native2_pread,
     .yapio_sys_pwrite = (pwritef)ime_client_native2_pwrite};
#endif

#define YAPIO_SYS_CALL(__sys_call)              \
    yapioSysCallOps->yapio_sys_ ## __sys_call

/* YAPIO_NIOVA function implementations are defined below, after yapio_exit()
 * and yapioTestGroups are declared (see "Niova IO helpers" section). */

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
    int    ybm_owner_rank_fpp;  //fpp only, file to which contents belong
    unsigned long ybm_blk_number : 62;      //number of the block - should not change!
    unsigned long ybm_not_hole : 1;   // block is empty (no data)
    unsigned long ybm_skip_io : 1;     // instruction to do IO or not
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
    bool                     ytc_stonewalled;
    int                      ytc_num_ops_expected;
    int                      ytc_num_ops_completed_before_stonewall;
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
    char                     ytc_suffix[YAPIO_MKSTEMP_TEMPLATE_LEN + 1];
    struct yapio_test_group *ytc_group;
    unsigned int             ytc_sparse_io;      //number of IOs skipped
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
    bool             ytg_keep_file;
    bool             ytg_leader_rank;
    bool             ytg_restart_from_previous_job;
    int              ytg_last_writer_ctx;
    char             ytg_suffix[YAPIO_MKSTEMP_TEMPLATE_LEN + 1];
    char             ytg_prefix[YAPIO_RECIPE_PARAM_STRLEN_MAX + 1];
    char             ytg_target_dir[PATH_MAX + 1];
} yapio_test_group_t;

static yapio_test_group_t *yapioMyTestGroup;
static yapio_test_group_t  yapioTestGroups[YAPIO_NUM_TEST_CTXS_MAX];
static int                 yapioNumTestGroups;

static int
yapio_relative_rank_get(const yapio_test_group_t *ytg, int idx_shift)
{
    int rank = (yapioMyRank - ytg->ytg_first_rank + idx_shift) %
        ytg->ytg_num_ranks;

    return rank < 0 ? (ytg->ytg_num_ranks - 1) : rank;
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

/* Ensure yapio's log_msg takes precedence over any prior definition
 * (e.g. niova log.h may define log_msg as LOG_MSG when YAPIO_NIOVA is set). */
#ifdef log_msg
#undef log_msg
#endif
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

        if (yapioIOErrorCnt)
            MPI_Abort(MPI_COMM_WORLD, -EIO);

        else
            MPI_Finalize();
    }

    exit(exit_rc);
}

#ifdef YAPIO_NIOVA
/* ---------------------------------------------------------------------------
 * Niova async IO — iopm-callback event loop
 *
 * yapio_niova_queue_work_cb() is registered as the iopm runtime callback via
 * niova_block_client_set_private_opts().  It runs on the iopm worker thread
 * already present inside the niova client.
 *
 * yapio_niova_completion_cb() is the per-IO completion callback passed to
 * NiovaBlockClientReadv/Writev.  It also fires on the same iopm worker thread.
 *
 * Because both functions share the same thread, all queue state (CIRCLEQ,
 * counters) is accessed without any mutex.  The only cross-thread
 * synchronisation is a single mutex+condvar used to notify the MPI main
 * thread when all blocks in a yapio_perform_io() call have completed.
 *
 * This mirrors the nbti_queue_work() / nbti_completion_cb() pattern in
 * niova-block-test.c client mode, adapted for yapio's MPI main-thread model.
 * -------------------------------------------------------------------------*/

#define YAPIO_NIOVA_DEF_QUEUE_DEPTH   12
#define YAPIO_NIOVA_MAX_QUEUE_DEPTH  256

static size_t yapioNiovaQueueDepth = YAPIO_NIOVA_DEF_QUEUE_DEPTH;
static char  *yapioNiovaSlotBufs   = NULL;

typedef struct yapio_niova_slot {
    struct iovec              yns_iov;    /* full-block iov, set once per submission */
    char                     *yns_buf;   /* per-slot buffer (blk_sz bytes)          */
    const yapio_blk_md_t     *yns_md;   /* block metadata (read-only)              */
    bool                      yns_running;
    CIRCLEQ_ENTRY(yapio_niova_slot) yns_lentry;
} yapio_niova_slot_t;

CIRCLEQ_HEAD(yapio_niova_slot_queue, yapio_niova_slot);

typedef struct {
    /* Slot pool — accessed only on the iopm worker thread, no mutex needed */
    yapio_niova_slot_t           *ynqs_slots;      /* dynamically allocated          */
    struct yapio_niova_slot_queue ynqs_idle;
    struct yapio_niova_slot_queue ynqs_running;
    int                           ynqs_num_ioh;    /* min(total_blocks, queue_depth) */
    int                           ynqs_ioh_in_progress;

    /* Work items — set by main thread before iopm starts, read-only after */
    const yapio_blk_md_t         *ynqs_md_array;
    int                           ynqs_total_blocks;
    int                           ynqs_next_j;     /* next md_array index to submit  */
    int                           ynqs_target;     /* completions needed before done  */
    size_t                        ynqs_blk_sz;
    bool                          ynqs_is_read;
    bool                          ynqs_ready;      /* main thread sets true to start  */
    yapio_test_ctx_t             *ynqs_ytc;        /* for stonewall + completed count */

    /* Results — written on iopm thread, read by main thread after done */
    int                           ynqs_ncompleted;
    int                           ynqs_error;

    /* Main thread notification — the only cross-thread synchronisation */
    pthread_mutex_t               ynqs_done_mutex;
    pthread_cond_t                ynqs_done_cond;
    bool                          ynqs_done;
} yapio_niova_queue_state_t;

static yapio_niova_queue_state_t yapioQueueState;

/* Forward declarations — these functions are defined later in the file. */
static int   yapio_verify_contents_of_io_buffer(const char *buf, size_t buf_len,
                                                 const yapio_blk_md_t *md);
static off_t yapio_get_rw_offset(const yapio_blk_md_t *md, size_t blk_sz);
static void  yapio_apply_contents_to_io_buffer(char *buf, size_t buf_len,
                                                const yapio_blk_md_t *md);

/* Signal the MPI main thread that all IO for this test context is done.
 * Called from the iopm worker thread (under no lock).
 */
static void
yapio_niova_signal_done(yapio_niova_queue_state_t *state)
{
    pthread_mutex_lock(&state->ynqs_done_mutex);
    state->ynqs_done = true;
    pthread_cond_signal(&state->ynqs_done_cond);
    pthread_mutex_unlock(&state->ynqs_done_mutex);
}

/* Completion callback — runs on the iopm worker thread.
 * Fires once per block; nclient_co handles chunk boundaries internally.
 * No mutex needed for CIRCLEQ or counter operations.
 */
static void
yapio_niova_completion_cb(void *arg, ssize_t rc)
{
    yapio_niova_slot_t        *slot  = (yapio_niova_slot_t *)arg;
    yapio_niova_queue_state_t *state = &yapioQueueState;

    if (rc < 0)
    {
        log_msg(YAPIO_LL_ERROR, "niova %s cb rc=%zd",
                state->ynqs_is_read ? "READ" : "WRITE", rc);
        if (!state->ynqs_error)
            state->ynqs_error = (int)rc;
    }
    else if (state->ynqs_is_read && yapioVerifyRead && !state->ynqs_error)
    {
        int vrc = yapio_verify_contents_of_io_buffer(slot->yns_buf,
                                                     state->ynqs_blk_sz,
                                                     slot->yns_md);
        if (vrc && !state->ynqs_error)
            state->ynqs_error = vrc;
    }

    slot->yns_running = false;
    CIRCLEQ_REMOVE(&state->ynqs_running, slot, yns_lentry);
    CIRCLEQ_INSERT_TAIL(&state->ynqs_idle, slot, yns_lentry);
    state->ynqs_ioh_in_progress--;
    state->ynqs_ncompleted++;
    state->ynqs_ytc->ytc_num_ops_completed_before_stonewall++;

    log_msg(YAPIO_LL_DEBUG, "niova %s done %d/%d err=%d",
            state->ynqs_is_read ? "READ" : "WRITE",
            state->ynqs_ncompleted, state->ynqs_target,
            state->ynqs_error);

    if (state->ynqs_ncompleted >= state->ynqs_target)
        yapio_niova_signal_done(state);
}

/* iopm runtime callback — runs on the iopm worker thread on every iteration.
 * Mirrors nbti_queue_work() in niova-block-test.c client mode.
 * No mutex needed: both this function and yapio_niova_completion_cb run on
 * the same iopm worker thread.
 */
static void
yapio_niova_queue_work_cb(void *arg)
{
    yapio_niova_queue_state_t *state = (yapio_niova_queue_state_t *)arg;

    if (!state->ynqs_ready || state->ynqs_done)
        return;

    /* Stonewall: stop submitting new blocks.  In-flight ops continue until
     * their completion callbacks fire.  Update target so the completion
     * callback knows when all submitted work is done.
     */
    if (yapioStoneWalled && !state->ynqs_ytc->ytc_stonewalled)
    {
        state->ynqs_ytc->ytc_stonewalled = 1;
        state->ynqs_target = state->ynqs_next_j;
        if (state->ynqs_ncompleted >= state->ynqs_target)
            yapio_niova_signal_done(state);
        return;
    }

    if (state->ynqs_ytc->ytc_stonewalled)
        return;

    int remaining = state->ynqs_next_j < state->ynqs_total_blocks
                    ? state->ynqs_total_blocks - state->ynqs_next_j : 0;
    if (remaining == 0)
        return;

    int to_submit = MIN(remaining,
                        state->ynqs_num_ioh - state->ynqs_ioh_in_progress);

    for (int i = 0; i < to_submit; i++)
    {
        if (CIRCLEQ_EMPTY(&state->ynqs_idle))
            break;

        const yapio_blk_md_t *md = &state->ynqs_md_array[state->ynqs_next_j];
        state->ynqs_next_j++;

        /* Skip flagged blocks without consuming a slot */
        if (md->ybm_skip_io)
        {
            state->ynqs_ncompleted++;
            state->ynqs_ytc->ytc_num_ops_completed_before_stonewall++;
            to_submit++; /* allow one more iteration to fill the gap */
            if (state->ynqs_ncompleted >= state->ynqs_target)
                yapio_niova_signal_done(state);
            continue;
        }

        off_t off = yapio_get_rw_offset(md, state->ynqs_blk_sz);
        if (off < 0)
        {
            log_msg(YAPIO_LL_ERROR, "yapio_get_rw_offset() failed j=%d",
                    state->ynqs_next_j - 1);
            if (!state->ynqs_error)
                state->ynqs_error = -ERANGE;
            break;
        }

        yapio_niova_slot_t *slot = CIRCLEQ_FIRST(&state->ynqs_idle);
        CIRCLEQ_REMOVE(&state->ynqs_idle, slot, yns_lentry);

        slot->yns_md           = md;
        slot->yns_iov.iov_base = slot->yns_buf;
        slot->yns_iov.iov_len  = state->ynqs_blk_sz;
        slot->yns_running      = true;

        vdev_vblk_t vblk = (vdev_vblk_t)(off / YAPIO_NIOVA_BLOCK_SIZE);

        if (!state->ynqs_is_read)
            yapio_apply_contents_to_io_buffer(slot->yns_buf,
                                               state->ynqs_blk_sz, md);

        CIRCLEQ_INSERT_TAIL(&state->ynqs_running, slot, yns_lentry);
        state->ynqs_ioh_in_progress++;

        log_msg(YAPIO_LL_DEBUG, "niova %s submit j=%d vblk=%lu bytes=%zu",
                state->ynqs_is_read ? "READ" : "WRITE",
                state->ynqs_next_j - 1,
                (uint64_t)vblk, state->ynqs_blk_sz);

        int rc = state->ynqs_is_read
            ? NiovaBlockClientReadv(yapioNiovaClient, vblk,
                                    &slot->yns_iov, 1,
                                    yapio_niova_completion_cb, slot)
            : NiovaBlockClientWritev(yapioNiovaClient, vblk,
                                     &slot->yns_iov, 1,
                                     yapio_niova_completion_cb, slot);

        if (rc < 0)
        {
            log_msg(YAPIO_LL_ERROR, "niova %s submit failed rc=%d j=%d",
                    state->ynqs_is_read ? "READ" : "WRITE",
                    rc, state->ynqs_next_j - 1);
            /* Drive through completion path to keep accounting correct */
            yapio_niova_completion_cb(slot, (ssize_t)rc);
        }
    }
}

/* ---------------------------------------------------------------------------
 * Vdev file parser
 *
 * File format (one entry per non-comment, non-blank line):
 *   <vdev_uuid>  [connect_string]
 *
 * Line number (0-based, skipping comments/blanks) maps to MPI rank.
 * connect_string (e.g. "unix:<target_uuid>") is optional per line; if
 * absent, yapioNiovaConnectStr (set via -C) is used as a fallback.
 * -------------------------------------------------------------------------*/
static int
yapio_niova_parse_vdev_file(void)
{
    FILE *f = fopen(yapioNiovaVdevFile, "r");
    if (!f)
    {
        log_msg(YAPIO_LL_FATAL, "fopen(%s): %s",
                yapioNiovaVdevFile, strerror(errno));
        return -errno;
    }

    char line[512];
    int  line_num = 0;
    int  rc       = -ENOENT;

    while (fgets(line, sizeof(line), f))
    {
        /* Skip comment and blank lines without counting them. */
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r')
            continue;

        if (line_num == yapioMyRank)
        {
            char vdev_str[UUID_STR_LEN + 1] = {0};
            char conn_str[256]              = {0};

            int n = sscanf(line, "%36s %255s", vdev_str, conn_str);
            if (n < 1)
            {
                log_msg(YAPIO_LL_FATAL,
                        "rank %d: malformed line %d in %s",
                        yapioMyRank, line_num, yapioNiovaVdevFile);
                fclose(f);
                return -EINVAL;
            }

            if (uuid_parse(vdev_str, yapioNiovaVdevUuid))
            {
                log_msg(YAPIO_LL_FATAL,
                        "rank %d: invalid UUID '%s' on line %d of %s",
                        yapioMyRank, vdev_str, line_num, yapioNiovaVdevFile);
                fclose(f);
                return -EINVAL;
            }

            /* Per-line connect string overrides the global -C value. */
            if (n == 2)
                strncpy(yapioNiovaConnectStr, conn_str,
                        sizeof(yapioNiovaConnectStr) - 1);

            rc = 0;
        }

        line_num++;
    }

    fclose(f);

    if (rc)
    {
        log_msg(YAPIO_LL_FATAL,
                "rank %d: no vdev entry found (file has %d entries, "
                "need at least %d): %s",
                yapioMyRank, line_num, yapioNumRanks, yapioNiovaVdevFile);
        return rc;
    }

    if (yapio_global_leader_rank() && line_num != yapioNumRanks)
        log_msg(YAPIO_LL_WARN,
                "vdev file has %d entries but MPI has %d ranks",
                line_num, yapioNumRanks);

    return 0;
}

/* Setup one niova block client per MPI rank, connected to the vdev UUID
 * assigned to this rank in the vdev file.  Must be called before
 * yapio_setup_buffers() so that ytg_blk_sz is already set to 4096.
 */
static void
yapio_niova_setup_clients(void)
{
    int rc = yapio_niova_parse_vdev_file();
    if (rc)
        yapio_exit(YAPIO_EXIT_ERR);

    if (!yapioNiovaConnectStr[0])
    {
        log_msg(YAPIO_LL_FATAL,
                "rank %d: no niova connect string "
                "(use -C <connect_str> or embed in vdev file)",
                yapioMyRank);
        yapio_exit(YAPIO_EXIT_ERR);
    }

    struct niova_block_client_xopts xopts = {0};

    /* "cp" means control-plane mode; anything else is a transport connect
     * string (e.g. "unix:<target_uuid>") parsed by the library. */
    bool cp_mode = (strcmp(yapioNiovaConnectStr, "cp") == 0);
    int  vdi_mode = cp_mode ? VDEV_MODE_CONTROL_PLANE : VDEV_MODE_CLIENT_TEST;

    if (!cp_mode)
    {
        rc = niova_block_client_parse_target_opt_string(yapioNiovaConnectStr,
                                                        &xopts.npcx_opts);
        if (rc)
        {
            log_msg(YAPIO_LL_FATAL,
                    "rank %d: parse_target_opt_string('%s'): %s",
                    yapioMyRank, yapioNiovaConnectStr, strerror(-rc));
            yapio_exit(YAPIO_EXIT_ERR);
        }
    }

    uuid_copy(xopts.npcx_opts.vdev_uuid, yapioNiovaVdevUuid);
    uuid_generate(xopts.npcx_opts.client_uuid);

    /* For CP mode vdi_num_vblks must be 0 — the library overwrites it with the
     * value returned by the control plane (mirrors niova-block-test behaviour).
     * For CLIENT_TEST mode, pre-calculate from -z or n*blk_sz.
     */
    size_t vdi_num_vblks = 0;
    if (!cp_mode)
    {
        size_t io_range_bytes = yapioNiovaVdevSizeBytes > 0
                                ? yapioNiovaVdevSizeBytes
                                : MAX(yapioNumBlksPerRank * yapioBlkSz,
                                      YAPIO_NIOVA_DEFAULT_FILE_SIZE);
        vdi_num_vblks = io_range_bytes / YAPIO_NIOVA_BLOCK_SIZE;
    }
	if (yapioNetOnly)
	{
		xopts.npcx_opts.flags |= NIOVA_BLOCK_FLAGS_NET_ONLY;
	}

    struct vdev_info vdi = {
        .vdi_mode      = vdi_mode,
        .vdi_num_vblks = vdi_num_vblks,
    };

    /* Register yapio_niova_queue_work_cb as the iopm runtime callback so it
     * runs on the same iopm worker thread as completion callbacks, eliminating
     * the need for any mutex on the hot IO submission/completion path.
     * Mirrors: niova_block_client_set_private_opts(..., ut2_client_runtime_cb, nbti)
     * in niova-block-test.c client mode.
     */
    rc = niova_block_client_set_private_opts(&xopts, &vdi,
                                              yapio_niova_queue_work_cb,
                                              &yapioQueueState);
    if (rc)
    {
        log_msg(YAPIO_LL_FATAL,
                "rank %d: niova_block_client_set_private_opts(): %s",
                yapioMyRank, strerror(-rc));
        yapio_exit(YAPIO_EXIT_ERR);
    }

    xopts.npcx_opts.queue_depth = (unsigned int)yapioNiovaQueueDepth;

    rc = NiovaBlockClientNew(&yapioNiovaClient, &xopts.npcx_opts);
    if (rc || !yapioNiovaClient)
    {
        log_msg(YAPIO_LL_FATAL,
                "rank %d: NiovaBlockClientNew(): rc=%s client=%p",
                yapioMyRank, strerror(-rc), yapioNiovaClient);
        yapio_exit(YAPIO_EXIT_ERR);
    }

    /* In CP mode adopt the vdev size from the control plane unless -z
     * was explicitly given. Mirrors: ut2Opts.iopmo_file_size =
     * niova_block_client_vdev_size(client) in niova-block-test.c.
     */
    if (cp_mode && yapioNiovaVdevSizeBytes == 0)
    {
        ssize_t cp_sz = niova_block_client_vdev_size(yapioNiovaClient);
        if (cp_sz > 0)
            yapioNiovaVdevSizeBytes = (size_t)cp_sz;
    }

    /* Print vdev info obtained from the control plane after connect. */
    {
        char vdev_uuid_str[UUID_STR_LEN];
        uuid_unparse_lower(yapioNiovaVdevUuid, vdev_uuid_str);

        ssize_t vdev_sz = niova_block_client_vdev_size(yapioNiovaClient);
        int     max_xfer = niova_block_client_max_xfer_vblks(yapioNiovaClient);

        yapioNiovaMaxXferVblks = max_xfer;

        uint64_t total_vblks   = (vdev_sz > 0) ?
                                  (uint64_t)vdev_sz / YAPIO_NIOVA_BLOCK_SIZE : 0;
        uint64_t total_chunks  = total_vblks >> VBLK_BITS;

        fprintf(stderr,
                "[rank %d] niova vdev connected:\n"
                "  vdev-uuid    : %s\n"
                "  connect      : %s\n"
                "  vdev-size    : %zd bytes (%.2f GiB)\n"
                "  total-vblks  : %lu  (%lu per chunk)\n"
                "  total-chunks : %lu  (chunk-size %u GiB)\n"
                "  max-xfer     : %d vblks (%d bytes)\n",
                yapioMyRank,
                vdev_uuid_str,
                yapioNiovaConnectStr,
                vdev_sz, (double)vdev_sz / (1ULL << 30),
                total_vblks, (uint64_t)VBLKS_PER_CHUNK,
                total_chunks, (unsigned)(VDEV_CHUNK_SIZE_BYTES >> 30),
                max_xfer, max_xfer * YAPIO_NIOVA_BLOCK_SIZE);

        if (!cp_mode && xopts.npcx_opts.net_target_addr[0])
            fprintf(stderr,
                    "  target-addr  : %s:%u\n",
                    xopts.npcx_opts.net_target_addr,
                    xopts.npcx_opts.net_target_port);
    }

    /* Validate that the user's -b block size is a multiple of the niova
     * 4K vblk size.  Do not override it — the user controls IO size via -b.
     */
    for (int i = 0; i < yapioNumTestGroups; i++)
    {
        if (yapioTestGroups[i].ytg_blk_sz % YAPIO_NIOVA_BLOCK_SIZE != 0)
        {
            log_msg(YAPIO_LL_FATAL,
                    "niova mode: block size %zu must be a multiple of %d",
                    yapioTestGroups[i].ytg_blk_sz, YAPIO_NIOVA_BLOCK_SIZE);
            yapio_exit(YAPIO_EXIT_ERR);
        }
    }
}

static void
yapio_niova_teardown_clients(void)
{
    if (yapioNiovaClient)
    {
        NiovaBlockClientDestroy(yapioNiovaClient);
        yapioNiovaClient = NULL;
    }
}
#endif /* YAPIO_NIOVA */

static int
yapio_mmap_open(const char *file, int flags, mode_t mode)
{
    int fd = open(file, flags, mode);
    if (fd >= YAPIO_FD_MAP_SIZE_MAX)
    {
        log_msg(YAPIO_LL_FATAL, "fd=%d exceeds map table size %u",
                fd, YAPIO_FD_MAP_SIZE_MAX);
        close(fd);
        return -ERANGE;
    }

    int prot = mode == O_RDONLY ? PROT_READ : (PROT_READ | PROT_WRITE);

    size_t len = (yapioMyTestGroup->ytg_num_blks_per_rank *
                  yapioMyTestGroup->ytg_blk_sz *
                  (yapioMyTestGroup->ytg_file_per_process ?
                   1 : yapioMyTestGroup->ytg_num_ranks));

    void *addr = mmap(NULL, len, prot, MAP_SHARED, fd, 0);
    if (addr == NULL)
    {
        const int error = errno;
        log_msg(YAPIO_LL_FATAL, "mmap() prot=%d len=%zd: %s",
                prot, len, strerror(error));
        close(fd);
        return -error;
    }

    log_msg(YAPIO_LL_DEBUG, "fd=%d addr=%p len=%zd", fd, addr, len);

    yapioFdMap[fd].yfm_fd = fd;
    yapioFdMap[fd].yfm_addr = addr;

    return fd;
}

static int
yapio_mmap_close(int fd)
{
    log_msg(YAPIO_LL_DEBUG, "fd=%d addr=%p", fd, yapioFdMap[fd].yfm_addr);

    size_t len = (yapioMyTestGroup->ytg_num_blks_per_rank *
                  yapioMyTestGroup->ytg_blk_sz);

    int rc = munmap(yapioFdMap[fd].yfm_addr, len);
    if (rc)
        log_msg(YAPIO_LL_ERROR, "munmap(): %s", strerror(errno));

    rc = close(yapioFdMap[fd].yfm_fd);
    if (rc)
        log_msg(YAPIO_LL_ERROR, "close(): %s", strerror(errno));

    return rc;
}

static int
yapio_mmap_fsync(int fd)
{
#if 1
    size_t len = (yapioMyTestGroup->ytg_num_blks_per_rank *
                  yapioMyTestGroup->ytg_blk_sz *
                  (yapioMyTestGroup->ytg_file_per_process ?
                   1 : yapioMyTestGroup->ytg_num_ranks));

    int rc = msync(yapioFdMap[fd].yfm_addr, len, MS_SYNC);
    if (rc)
    {
        rc = -errno;

        log_msg(YAPIO_LL_WARN, "fd=%d addr=%p rc=%d",
                fd, yapioFdMap[fd].yfm_addr, rc);
    }
#else
    int rc = fsync(fd);
#endif
    return rc;
}

static ssize_t
yapio_mmap_pread(int fd, void *buf, size_t count, off_t offset)
{
    const char *addr = yapioFdMap[fd].yfm_addr;

    memcpy((char *)buf, &addr[offset], count);

    return count;
}

static ssize_t
yapio_mmap_pwrite(int fd, const void *buf, size_t count, off_t offset)
{
    char *addr = yapioFdMap[fd].yfm_addr;

    log_msg(YAPIO_LL_DEBUG, "%p off=%ld cnt=%zd",
            &addr[offset], offset, count);

    memcpy(&addr[offset], (char *)buf, count);

    return count;
}

/* Timer related calls.
 */
#define YAPIO_NSEC_PER_SEC 1000000000L
#define YAPIO_USEC_PER_SEC 1000000L

#define YAPIO_TIME_PRINT_SPEC "%.3f"
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
                "\t-m  I/O Mode - \n"
                "\t    - P (posix (default))\n"
                "\t    - I (IME native)\n"
                "\t    - m (mmap)\n"
                "\t-n  Network only test\n"
                "\t-N  Number of blocks per task\n"
                "\t-V  Disable read verification\n"
                "\t-W  Enable word-level buffer uniqueness (default: fast seed fill)\n"
                "\t-q  Niova queue depth (default 12, max 256, niova mode only)\n"
                "\t-s  Display test duration and barrier wait times\n"
                "\t-S  Number of seconds before stonewalling\n\n"
                "\t-t  Test description\n"
                "\t    - Prefix:     P{prefix}\n"
                "\t    - Restart:    X{suffix}\n"
                "\t    - Pattern:    sequential (s), random (R), strided (S)\n"
                "\t    - I/O Op:     read (r), write (w)\n"
                "\t    - Locality:   local (L), distributed (D)\n"
                "\t    - Sparse I/O: missed percentage (M), only do M percent of I/Os\n"
                "\t    - Options:    backwards (b), holes (h), no-fsync (f)\n"
                "\t    - Parameters: block-size (B), blocks-per-rank (n),\n"
                "\t                  num-ranks (N), file-per-process (F)\n"
                "\t                  keep-file (K)\n"
                "\n\t    Example: -t wsL,rRD\n"
                "\t      sequential write, distribute reads randomly\n"
                "\n\t    Example: -t K:F:wsL\n"
                "\t      Use file-per-process and keep output files.\n"
                "\n\t    Example: -t wsL,rRD -t N4:B4096:F:ws,rsb\n"
                "\t      Run two tests simultaneously\n"
                "\n\t    Example: -t n10:wRDM.9\n"
                "\t      Run 9 random distributed writes over 10 blocks\n"
                "\t      (1 hole is created)\n"
                "\n\t    Example: -t Pmy-test-name:wsL,rRD\n"
                "\t      Prefix output files with 'my-test-name'\n"
                "\n\t    Example: -t Xabc123:rRD,rsL,rSD,wRD\n"
                "\t      Restart from previous yapio instance 'abc123'\n\n",
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

static int
yapio_test_recipe_param_gather(const char *param_str, char *buf)
{
    int j;

    for (j = 0; j < YAPIO_RECIPE_PARAM_STRLEN_MAX - 1 && param_str[j] != ':';
         j++)
        buf[j] = param_str[j];

    if (param_str[j] != ':')
    {
        if (j == YAPIO_RECIPE_PARAM_STRLEN_MAX)
        {
            log_msg(YAPIO_LL_ERROR, "recipe input string is too long");
            return -E2BIG;
        }
        return -EINVAL;
    }

    buf[j] = '\0';

    return j + 1; //include ':' char
}

static int
yapio_test_recipe_param_to_ull(const char *param_str, size_t *result)
{
    char recipe_param_tmp_str[YAPIO_RECIPE_PARAM_STRLEN_MAX];

    int len = yapio_test_recipe_param_gather(param_str, recipe_param_tmp_str);

    if (result)
        *result = strtoull(recipe_param_tmp_str, NULL, 10);

    log_msg_r0(YAPIO_LL_DEBUG, "%s %zu", recipe_param_tmp_str, *result);

    return len;
}

static int
yapio_test_recipe_param_to_string(const char *param_str, char *dest)
{
    char recipe_param_tmp_str[YAPIO_RECIPE_PARAM_STRLEN_MAX];

    int len = yapio_test_recipe_param_gather(param_str, recipe_param_tmp_str);

    strncpy(dest, recipe_param_tmp_str, len);

    log_msg_r0(YAPIO_LL_DEBUG, "%s", dest);

    return len;
}

static void
yapio_test_group_init(yapio_test_group_t *ytg)
{
    ytg->ytg_num_contexts = 1;
    ytg->ytg_num_blks_per_rank = yapioNumBlksPerRank;
    ytg->ytg_blk_sz = yapioBlkSz;
    ytg->ytg_file_per_process = false;
    ytg->ytg_restart_from_previous_job = false;
    ytg->ytg_group_num = yapioNumTestGroups;
    strncpy(ytg->ytg_prefix, YAPIO_DEFAULT_FILE_PREFIX,
            YAPIO_RECIPE_PARAM_STRLEN_MAX);
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
        case 'P':
            i += yapio_test_recipe_param_to_string(&recipe_str[i + 1],
                                                   ytg->ytg_prefix);
            break;
        case 'X':
            i += yapio_test_recipe_param_to_string(&recipe_str[i + 1],
                                                   ytg->ytg_suffix);
            ytg->ytg_restart_from_previous_job = true;
            break;
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
        case 'M':
            if (recipe_str[i + 1] != '.')
            {
                printf("Percentage should start by '.'\n");
                yapio_print_help(YAPIO_EXIT_ERR);
            }
            /* find where percentage stops */
            int len_pdot = strlen("M.");
            i += len_pdot;
            unsigned int end = i;
            while (end < recipe_str_len)
            {
                if (!isdigit(recipe_str[end]))
                    break;
                end++;
            }
            unsigned long int percent = 0;
            percent = strtoull(&recipe_str[i], NULL, 10);

            /* count number of skipped IOs */
            ytc->ytc_sparse_io = percent * ytg->ytg_num_blks_per_rank;
            while (i < end)
            {
                ytc->ytc_sparse_io = ytc->ytc_sparse_io / 10;
                i++;
            }
            printf("skip %d IOs\n", ytc->ytc_sparse_io);

            i = end - 1;
            break;
        case 'K':
            ytg->ytg_keep_file = true;
            i++;
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

    int *ranks = YAPIO_CALLOC(ytg->ytg_num_ranks, sizeof(int));
    if (!ranks)
        return -errno;

    int i;
    for (i = 0; i < ytg->ytg_num_ranks; i++)
        ranks[i] = start_rank + i;

    rc = MPI_Group_incl(group_world, ytg->ytg_num_ranks, ranks,
                        &ytg->ytg_group);

    YAPIO_FREE(ranks);

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

static const yapio_io_syscall_ops_t *
yapio_parse_io_mode(const char *io_mode_str)
{
    if (strnlen(io_mode_str, 2) > 1)
        return NULL;

    switch (io_mode_str[0])
    {
    case 'P':
        return yapio_io_modes[YAPIO_IO_MODE_DEFAULT];

    case 'I':
        yapioModeCurrent = YAPIO_IO_MODE_IME;
        yapioTestFileNamePrefix = "ime://";
        return yapio_io_modes[YAPIO_IO_MODE_IME];

    case 'M':
        yapioModeCurrent = YAPIO_IO_MODE_MPIIO;
        yapioTestFileNamePrefix = "XXX://";
        return yapio_io_modes[YAPIO_IO_MODE_MPIIO];

    case 'm':
        yapioModeCurrent = YAPIO_IO_MODE_MMAP;
        return yapio_io_modes[YAPIO_IO_MODE_MMAP];

#ifdef YAPIO_NIOVA
    case 'N':
        yapioModeCurrent = YAPIO_IO_MODE_NIOVA;
        return NULL; // niova does not use the syscall ops table
#endif

    default:
        return NULL;
    }

    return NULL;
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
            if (yapioBlkSz == 0 || yapioBlkSz % YAPIO_NIOVA_BLOCK_SIZE != 0)
            {
                fprintf(stderr, "Block size must be a non-zero multiple of %d.\n",
                        YAPIO_NIOVA_BLOCK_SIZE);
                yapio_print_help(YAPIO_EXIT_ERR);
            }
            break;
        case 'D':
            yapioDecomposeCnt = MIN(atoi(optarg), YAPIO_DECOMPOSE_MAX);
            break;
        case 'd':
            yapioDbgLevel = MIN(atoi(optarg), YAPIO_LL_MAX);
            break;
        case 'h':
            yapio_print_help(YAPIO_EXIT_OK);
            break;
        case 'm':
            yapioSysCallOps = yapio_parse_io_mode(optarg);
            break;
        case 'V':
            yapioVerifyRead = false;
            break;
        case 'W':
            yapioWordLevelVerify = true;
            break;
        case 'N':
            yapioNumBlksPerRank = strtoull(optarg, NULL, 10);
            break;
        case 'n':
            yapioNetOnly = true;
            break;
        case 'P':
            yapioPolluteBlks = true;
            break;
        case 's':
            yapioDisplayStats = true;
            break;
        case 'S':
            yapioUseStoneWalling = true;
            yapioStoneWallNsecs = strtoull(optarg, NULL, 10);
            if (yapioStoneWallNsecs <= 0)
            {
                fprintf(stderr, "Invalid stonewalling value.\n");
                yapio_print_help(YAPIO_EXIT_ERR);
            }
            break;
        case 't':
            if (yapio_parse_test_recipe(optarg))
                yapio_print_help(YAPIO_EXIT_ERR);

            yapioNumTestGroups++;
            break;
#ifdef YAPIO_NIOVA
        case 'v':
            yapioNiovaVdevFile = optarg;
            break;
        case 'C':
            strncpy(yapioNiovaConnectStr, optarg,
                    sizeof(yapioNiovaConnectStr) - 1);
            break;
        case 'z':
            yapioNiovaVdevSizeBytes = strtoull(optarg, NULL, 10);
            break;
        case 'q':
            yapioNiovaQueueDepth =
                MIN((size_t)strtoull(optarg, NULL, 10),
                    (size_t)YAPIO_NIOVA_MAX_QUEUE_DEPTH);
            if (yapioNiovaQueueDepth == 0)
                yapioNiovaQueueDepth = YAPIO_NIOVA_DEF_QUEUE_DEPTH;
            break;
#endif
        default:
            yapio_print_help(YAPIO_EXIT_ERR);
            break;
        }
    }

    /* Check for user provided test directory parameter which should be at the
     * end.  Not required in niova mode since IO goes directly to nisd.
     */
    yapioTestRootDir = argv[optind];
    if (yapioModeCurrent != YAPIO_IO_MODE_NIOVA)
    {
        if (!yapioTestRootDir || argc > (optind + 1))
            yapio_print_help(YAPIO_EXIT_ERR);
    }

#ifdef YAPIO_NIOVA
    if (yapioModeCurrent == YAPIO_IO_MODE_NIOVA && !yapioNiovaVdevFile)
    {
        log_msg(YAPIO_LL_FATAL,
                "niova mode (-m N) requires a vdev file (-v <path>)");
        yapio_print_help(YAPIO_EXIT_ERR);
    }
#endif

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
    int rc = mkdir(yapioTestRootDir, 0740);
    if (rc == 0)
        return;

    if (errno != EEXIST)
        log_msg(YAPIO_LL_FATAL, "mkdir(%s):  %s", yapioTestRootDir,
                strerror(errno));

    struct stat stb;
    rc = stat(yapioTestRootDir, &stb);
    if (rc)
        rc = errno;

    else if (!S_ISDIR(stb.st_mode))
        rc = ENOTDIR;

    if (rc)
        log_msg(YAPIO_LL_FATAL, "%s:  %s", yapioTestRootDir, strerror(rc));
}

static int
yapio_make_filename(char *file_name, int rank)
{
    int rc = rank >= 0 ?
        snprintf(file_name, PATH_MAX, "%s%s.%d",
                 yapioTestFileNamePrefix, yapioTestFileName, rank) :
        snprintf(file_name, PATH_MAX, "%s%s",
                 yapioTestFileNamePrefix, yapioTestFileName);

    if (rc > PATH_MAX)
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

    int rc = yapio_make_filename(file_per_process_name, rank);
    if (rc)
        return rc;

    if (yapioFileDescFpp[rank] < 0)
    {
        yapioFileDescFpp[rank] =
            YAPIO_SYS_CALL(open)(file_per_process_name, oflags, 0644);

        if (yapioFileDescFpp[rank] < 0)
        {
            rc = -errno;
            log_msg(YAPIO_LL_ERROR, "open `%s': %s",
                    file_per_process_name, strerror(errno));
        }
        else
        {
            log_msg(YAPIO_LL_DEBUG, "%s", file_per_process_name);

            if (yapioModeCurrent == YAPIO_IO_MODE_MMAP)
            {
                size_t len = (yapioMyTestGroup->ytg_num_blks_per_rank *
                              yapioMyTestGroup->ytg_blk_sz);

                if (ftruncate(yapioFileDescFpp[rank], len))
                {
                    log_msg(YAPIO_LL_FATAL, "ftruncate(): %s",
                            strerror(errno));
                }
            }
        }
    }

    return rc;
}


/**
 * yapio_setup_test_file - Rank0 will create a temp file and broadcast
 *    the name the to the other ranks who will then also open the temp file.
 */
static void
yapio_setup_test_file(yapio_test_group_t *ytg)
{
    yapio_verify_test_directory();

    int path_len = snprintf(yapioTestFileName, PATH_MAX, "%s/%s%s",
                            yapioTestRootDir, ytg->ytg_prefix,
                            YAPIO_MKSTEMP_TEMPLATE);

    /* check if output got truncated */
    if (path_len == -1)
        log_msg(YAPIO_LL_FATAL, "File name got truncated: %s",
                strerror(errno));

    if (path_len > PATH_MAX)
        log_msg(YAPIO_LL_FATAL, "%s", strerror(ENAMETOOLONG));

    if (!ytg->ytg_restart_from_previous_job)
    {
        if (yapio_leader_rank())
        {
            yapioFileDesc = mkstemp(yapioTestFileName);
            if (yapioFileDesc < 0)
                log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));

            if (yapioModeCurrent == YAPIO_IO_MODE_MMAP &&
                !ytg->ytg_file_per_process)
            {
                size_t len = (ytg->ytg_num_blks_per_rank *
                              ytg->ytg_blk_sz *
                              ytg->ytg_num_ranks);

                if (ftruncate(yapioFileDesc, len))
                {
                    log_msg(YAPIO_LL_FATAL, "ftruncate(): %s",
                            strerror(errno));
                }
            }

            /* File will be reopened below by all ranks.
             */
            close(yapioFileDesc);

            strncpy(ytg->ytg_suffix,
                    &yapioTestFileName[path_len - YAPIO_MKSTEMP_TEMPLATE_LEN],
                    YAPIO_MKSTEMP_TEMPLATE_LEN);

            log_msg(YAPIO_LL_DEBUG, "%s", yapioTestFileName);
        }
        /* Broadcast only the section of the filename which was modified by
         * mkstemp().
         */
        MPI_OP_START;
        MPI_Bcast(ytg->ytg_suffix, YAPIO_MKSTEMP_TEMPLATE_LEN, MPI_CHAR, 0,
                  ytg->ytg_comm);
        MPI_OP_END;
    }


    strncpy(&yapioTestFileName[path_len - YAPIO_MKSTEMP_TEMPLATE_LEN],
            ytg->ytg_suffix, YAPIO_MKSTEMP_TEMPLATE_LEN + 1);

    if (ytg->ytg_file_per_process)
    {
        /* Remove the mkstemp file.
         */
        if (yapio_leader_rank())
            unlink(yapioTestFileName);

        yapioFileDescFpp = YAPIO_CALLOC(yapioNumRanks, sizeof(int));
        if (!yapioFileDescFpp)
            log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));

        int i;
        for (i = 0; i < yapioNumRanks; i++)
            yapioFileDescFpp[i] = -1;

        const int flags = ytg->ytg_restart_from_previous_job ?
            O_RDWR : (O_CREAT | O_EXCL | O_RDWR);

        int rc = yapio_open_fpp_file(yapio_relative_rank_get(ytg, 0), flags);
        if (rc)
            log_msg(YAPIO_LL_FATAL, "yapio_open_fpp_file: %s", strerror(-rc));

        yapio_make_filename(yapioTestFileNameFpp,
                            yapio_relative_rank_get(ytg, 0));

        log_msg(YAPIO_LL_DEBUG, "yapioTestFileNameFpp=%s",
                yapioTestFileNameFpp);
    }
    else
    {
        /* Re-make the filename so that mode specific prefixes may be applied.
         */
        char tmp_filename[PATH_MAX + 1];

        yapio_make_filename(tmp_filename, -1);

        strncpy(yapioTestFileName, tmp_filename, PATH_MAX + 1);

//        snprintf(file_name, PATH_MAX, "%s%s.%d",
//                 yapioTestFileNamePrefix, yapioTestFileName, rank) :

        log_msg(YAPIO_LL_DEBUG, "yapioTestFileName=%s tf=%s tf=%s",
                yapioTestFileName, yapioTestFileNamePrefix, yapioTestFileName);

        yapioFileDesc = YAPIO_SYS_CALL(open)(yapioTestFileName, O_RDWR, 0644);
        if (yapioFileDesc < 0)
            log_msg(YAPIO_LL_FATAL, "%s", strerror(errno));
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
                rc = YAPIO_SYS_CALL(close)(yapioFileDescFpp[i]);
                if (rc < 0)
                    break;
            }
        }
    }
    else
    {
        rc = YAPIO_SYS_CALL(close)(yapioFileDesc);
    }

    if (rc < 0)
        log_msg(YAPIO_LL_FATAL, "close: %s", strerror(errno));
}

static void
yapio_destroy_buffers(void)
{
    if (yapioSourceBlkMd)
        YAPIO_FREE(yapioSourceBlkMd);

    if (yapioIOBuf)
        YAPIO_FREE(yapioIOBuf);

#ifdef YAPIO_NIOVA
    if (yapioQueueState.ynqs_slots)
    {
        YAPIO_FREE(yapioQueueState.ynqs_slots);
        yapioQueueState.ynqs_slots = NULL;
    }

    if (yapioNiovaSlotBufs)
    {
        YAPIO_FREE(yapioNiovaSlotBufs);
        yapioNiovaSlotBufs = NULL;
    }
#endif
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
    yapioSourceBlkMd = YAPIO_CALLOC(ytg->ytg_num_blks_per_rank,
                                    sizeof(yapio_blk_md_t));
    if (yapioSourceBlkMd == NULL)
        yapio_destroy_buffers_and_abort();

    yapioIOBuf = YAPIO_CALLOC(1, ytg->ytg_blk_sz);
    if (yapioIOBuf == NULL)
        yapio_destroy_buffers_and_abort();

#ifdef YAPIO_NIOVA
    if (yapioModeCurrent == YAPIO_IO_MODE_NIOVA)
    {
        /* Allocate the slot pool and per-slot IO buffers.
         * yapioNiovaQueueDepth is fixed after getopts, ytg_blk_sz is fixed
         * after test-group setup — both are known at this point.
         */
        yapioQueueState.ynqs_slots =
            YAPIO_CALLOC(yapioNiovaQueueDepth, sizeof(yapio_niova_slot_t));
        if (yapioQueueState.ynqs_slots == NULL)
            yapio_destroy_buffers_and_abort();

        yapioNiovaSlotBufs =
            YAPIO_CALLOC(yapioNiovaQueueDepth, ytg->ytg_blk_sz);
        if (yapioNiovaSlotBufs == NULL)
            yapio_destroy_buffers_and_abort();
    }
#endif
}

yapio_blk_md_t *
yapio_test_ctx_to_md_array(const yapio_test_ctx_t *,
                           enum yapio_test_ctx_mdh_in_out, int *);

static bool
skip_io_or_not(int ops_left, int skips_left)
{
    /* for random number generation */
    srand(time(NULL));

    /* skip randomly or skip all remaining IOs
     * because there's no room for choosing */
    if (skips_left &&
        (skips_left == ops_left || rand() % (ops_left / skips_left)))
        return true;
    else
        return false;
}

/* select skipped IOs */
static void
select_holes(int num_ops, int skips_left, yapio_blk_md_t *md)
{
    int i;
    for (i = 0; i < num_ops; i++)
    {
        if (skip_io_or_not(num_ops - i, skips_left))
        {
            log_msg(YAPIO_LL_DEBUG, "skip IO # %d\n", i);
            md[i].ybm_skip_io = 1;
            skips_left--;
        }
    }
}

static int
yapio_md_buffer_io(const yapio_test_group_t *ytg, const bool dump)
{
    int rc = snprintf(yapioRestartFileName, PATH_MAX, "%s/.%s%s.%d.md",
                      yapioTestRootDir, ytg->ytg_prefix, ytg->ytg_suffix,
                      yapio_relative_rank_get(ytg, 0));
    if (rc >= PATH_MAX || rc == -1)
    {
        log_msg(YAPIO_LL_ERROR, "Md filename is too long");
        return -ENAMETOOLONG;
    }

    const size_t nblks_per_rank = ytg->ytg_num_blks_per_rank;

    FILE *file = fopen(yapioRestartFileName, dump ? "w" : "r");
    if (!file)
    {
        int rc = -errno;
        log_msg(YAPIO_LL_ERROR, "fopen(`%s'): %s", yapioRestartFileName,
                strerror(errno));
        return rc;
    }

    yapio_blk_md_t *md_array;
    if (dump)
    {
        /* use metadata from last iteration */
        int test_ctx_idf = ytg->ytg_last_writer_ctx;
        const yapio_test_ctx_t *ytc = &ytg->ytg_contexts[test_ctx_idf];

        md_array = yapio_test_ctx_to_md_array(ytc, YAPIO_TEST_CTX_MDH_OUT,
                                              NULL);
        if (!md_array)
            md_array = yapio_test_ctx_to_md_array(ytc, YAPIO_TEST_CTX_MDH_IN,
                                                  NULL);
    }
    else
    {
        md_array = yapioSourceBlkMd;
    }

    log_msg(YAPIO_LL_DEBUG, "%p %d lc=%d", md_array, ytg->ytg_num_contexts,
            ytg->ytg_last_writer_ctx);

    size_t io_rc = dump ?
        fwrite(md_array, sizeof(yapio_blk_md_t), nblks_per_rank, file) :
        fread(md_array, sizeof(yapio_blk_md_t), nblks_per_rank, file);

    if (io_rc != nblks_per_rank)
    {
        int rc = -errno;
        log_msg(YAPIO_LL_ERROR, "%s(`%s'): %s (%zu / %zu)",
                dump ? "fwrite" : "fread", yapioRestartFileName,
                strerror(errno), io_rc, nblks_per_rank);

        return rc;
    }

    if (fclose(file))
    {
        int rc = -errno;
        log_msg(YAPIO_LL_ERROR, "fclose(`%s'): %s", yapioRestartFileName,
                strerror(errno));
        return rc;
    }

    return 0;
}

/* read input md file to initialize yapioMyRank's md array */
static void
yapio_initialize_source_md_buffer_from_file(const yapio_test_group_t *ytg)
{
    if (yapio_md_buffer_io(ytg, false))
        log_msg(YAPIO_LL_FATAL, "yapio_md_buffer_io() failed");

    /* rank 0 makes sure each rank can read its md file */
    int n = 1;
    int global_sum = 0;
    MPI_Reduce(&n, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (yapioMyRank == 0 && global_sum != yapioNumRanks)
    {
        printf("%d rank(s) unable to open metadata file\n",
               yapioNumRanks - global_sum);
        /* stop execution for all ranks */
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void
yapio_initialize_source_md_buffer(const yapio_test_group_t *ytg)
{
    if (ytg->ytg_restart_from_previous_job)
        return yapio_initialize_source_md_buffer_from_file(ytg);

    int rel_rank = yapio_relative_rank_get(ytg, 0);
    size_t i;

    for (i = 0; i < ytg->ytg_num_blks_per_rank; i++)
    {
        yapioSourceBlkMd[i].ybm_writer_rank = yapioMyRank;

        /* In niova mode each rank owns a separate vdev, so block numbers
         * always start at 0 (same as file-per-process semantics).
         * In shared-file modes the rank offset is applied so that each rank
         * writes to a distinct region of the shared file.
         *
         * When -z <vdev_size> is specified in niova mode, spread n IOs evenly
         * across the full vdev (stride = vdev_total_blks / n).  This mirrors
         * niova-block-test where -N (nops) and the address range are
         * independent: random IOs pick from [0, max_blk) regardless of -N.
         */
        bool fpp = ytg->ytg_file_per_process ||
                   (yapioModeCurrent == YAPIO_IO_MODE_NIOVA);
        size_t blk_num = fpp ? i : rel_rank * ytg->ytg_num_blks_per_rank + i;

#ifdef YAPIO_NIOVA
        if (yapioModeCurrent == YAPIO_IO_MODE_NIOVA &&
            yapioNiovaVdevSizeBytes > 0 &&
            ytg->ytg_num_blks_per_rank > 0)
        {
            size_t vdev_total_blks = yapioNiovaVdevSizeBytes / ytg->ytg_blk_sz;
            size_t stride = vdev_total_blks / ytg->ytg_num_blks_per_rank;
            if (stride < 1) stride = 1;
            blk_num = i * stride;
        }
#endif
        yapioSourceBlkMd[i].ybm_blk_number = blk_num;

        yapioSourceBlkMd[i].ybm_owner_rank_fpp =
            ytg->ytg_file_per_process ? rel_rank : 0;

        yapioSourceBlkMd[i].ybm_not_hole = 0;
    }
}

static void
yapio_source_md_update_writer_rank(size_t source_md_idx, int new_writer_rank)
{
    if (source_md_idx >= yapioMyTestGroup->ytg_num_blks_per_rank)
        log_msg(YAPIO_LL_FATAL, "out of bounds source_md_idx=%zu",
                source_md_idx);

    log_msg(YAPIO_LL_TRACE, "%d %zu", new_writer_rank, source_md_idx);

    yapioSourceBlkMd[source_md_idx].ybm_writer_rank = new_writer_rank;
    yapioSourceBlkMd[source_md_idx].ybm_not_hole = 1;
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

    if (yapioWordLevelVerify)
    {
        /* Word-level unique fill: each 8-byte word gets a distinct value
         * derived from block metadata + word offset.  Enables per-word
         * corruption detection but is slow for large block sizes.
         */
        size_t i;
        for (i = 0; i < num_words; i++)
        {
            buffer_of_longs[i] = yapio_get_content_word(md, i);
            log_msg(YAPIO_LL_TRACE, "%zu:%llx", i, buffer_of_longs[i]);
        }
    }
    else
    {
        /* Fast seed fill: compute one value per block from block metadata,
         * write it to the first word, then use exponential memcpy doubling
         * to broadcast it across the entire buffer.  O(log N) memcpy calls
         * let libc's SIMD-optimised memcpy do the heavy lifting.
         */
        unsigned long long seed =
            yapio_get_blk_magic(md->ybm_blk_number) +
            md->ybm_writer_rank +
            md->ybm_blk_number +
            md->ybm_owner_rank_fpp;

        buffer_of_longs[0] = seed;
        size_t filled = sizeof(unsigned long long);
        while (filled * 2 <= buf_len)
        {
            memcpy(buf + filled, buf, filled);
            filled *= 2;
        }
        if (filled < buf_len)
            memcpy(buf + filled, buf, buf_len - filled);

        log_msg(YAPIO_LL_TRACE, "blk=%lu seed=%llx", md->ybm_blk_number, seed);
    }
}

static int
yapio_verify_contents_of_io_buffer(const char *buf, size_t buf_len,
                                   const yapio_blk_md_t *md)
{
    if (!md->ybm_not_hole)
    {
        log_msg(YAPIO_LL_DEBUG, "No verification for hole");
        return 0;
    }

    const unsigned long long *buffer_of_longs = (unsigned long long *)buf;
    size_t num_words = buf_len / sizeof(unsigned long long);

    if (yapioWordLevelVerify)
    {
        size_t i;
        for (i = 0; i < num_words; i++)
        {
            if (buffer_of_longs[i] != yapio_get_content_word(md, i))
            {
                log_msg(YAPIO_LL_ERROR,
                        "blk=%lu word=%zu got=%llx expected=%llx",
                        md->ybm_blk_number, i, buffer_of_longs[i],
                        yapio_get_content_word(md, i));
                return -1;
            }
            log_msg(YAPIO_LL_TRACE, "OK %zu:%llx", i, buffer_of_longs[i]);
        }
    }
    else
    {
        /* Re-derive the seed used at write time and verify every word. */
        unsigned long long seed =
            yapio_get_blk_magic(md->ybm_blk_number) +
            md->ybm_writer_rank +
            md->ybm_blk_number +
            md->ybm_owner_rank_fpp;

        size_t i;
        for (i = 0; i < num_words; i++)
        {
            if (buffer_of_longs[i] != seed)
            {
                log_msg(YAPIO_LL_ERROR,
                        "blk=%lu word=%zu got=%llx expected=%llx",
                        md->ybm_blk_number, i, buffer_of_longs[i], seed);
                return -1;
            }
            log_msg(YAPIO_LL_TRACE, "OK %zu:%llx", i, buffer_of_longs[i]);
        }
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

            if (YAPIO_SYS_CALL(fsync)(yapioFileDescFpp[i]))
            {
                rc = errno;
                break;
            }
        }
    }
    else
    {
        if (YAPIO_SYS_CALL(fsync)(yapioFileDesc))
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

    const yapio_blk_md_t *md_array =
        yapio_test_ctx_to_md_array(ytc, YAPIO_TEST_CTX_MDH_IN,
                                   &ytc->ytc_num_ops_expected);

    log_msg(YAPIO_LL_DEBUG, "rank=%d op=%s blk_sz=%zu nops=%d",
            yapioMyRank, ytc->ytc_read ? "read" : "write",
            ytg->ytg_blk_sz, ytc->ytc_num_ops_expected);

#ifdef YAPIO_NIOVA
    if (yapioModeCurrent == YAPIO_IO_MODE_NIOVA)
    {
        /* Niova path: iopm-callback event loop.
         * The iopm worker thread inside yapioNiovaClient drives all IO via
         * yapio_niova_queue_work_cb() and yapio_niova_completion_cb().
         * The MPI main thread blocks here on a single condvar until done.
         */
        yapio_niova_queue_state_t *qs = &yapioQueueState;

        /* Preserve the slot array allocated in yapio_alloc_buffers() across
         * the reset — it is reused across multiple yapio_perform_io() calls.
         */
        yapio_niova_slot_t *saved_slots = qs->ynqs_slots;
        memset(qs, 0, sizeof(*qs));
        qs->ynqs_slots = saved_slots;

        pthread_mutex_init(&qs->ynqs_done_mutex, NULL);
        pthread_cond_init(&qs->ynqs_done_cond, NULL);

        CIRCLEQ_INIT(&qs->ynqs_idle);
        CIRCLEQ_INIT(&qs->ynqs_running);

        qs->ynqs_num_ioh      = MIN(ytc->ytc_num_ops_expected,
                                    (int)yapioNiovaQueueDepth);
        qs->ynqs_md_array     = md_array;
        qs->ynqs_total_blocks = ytc->ytc_num_ops_expected;
        qs->ynqs_target       = ytc->ytc_num_ops_expected;
        qs->ynqs_blk_sz       = ytg->ytg_blk_sz;
        qs->ynqs_is_read      = ytc->ytc_read;
        qs->ynqs_ytc          = ytc;

        ytc->ytc_num_ops_completed_before_stonewall = 0;

        /* Wire up per-slot IO buffers and populate the idle queue */
        for (size_t i = 0; i < yapioNiovaQueueDepth; i++)
        {
            qs->ynqs_slots[i].yns_buf =
                yapioNiovaSlotBufs + i * ytg->ytg_blk_sz;
            CIRCLEQ_INSERT_TAIL(&qs->ynqs_idle,
                                &qs->ynqs_slots[i], yns_lentry);
        }

        /* Setting ynqs_ready=true allows yapio_niova_queue_work_cb() —
         * already being called by the iopm worker thread — to start
         * submitting IO.  No barrier is needed: the iopm thread will pick
         * up the flag on its next iteration.
         */
        qs->ynqs_ready = true;

        /* MPI main thread waits here; all IO happens on the iopm thread. */
        pthread_mutex_lock(&qs->ynqs_done_mutex);
        while (!qs->ynqs_done)
            pthread_cond_wait(&qs->ynqs_done_cond, &qs->ynqs_done_mutex);
        pthread_mutex_unlock(&qs->ynqs_done_mutex);

        rc = qs->ynqs_error;

        pthread_mutex_destroy(&qs->ynqs_done_mutex);
        pthread_cond_destroy(&qs->ynqs_done_cond);

        return rc;
    }
#endif

    /* POSIX path: sequential j-loop. */
    const int print_interval = ytc->ytc_num_ops_expected > 10
                               ? ytc->ytc_num_ops_expected / 10 : 1;

    int j;
    for (j = 0, ytc->ytc_num_ops_completed_before_stonewall = 0;
         j < ytc->ytc_num_ops_expected;
         j++, ytc->ytc_num_ops_completed_before_stonewall++)
    {
        if (yapioStoneWalled)
        {
            log_msg(YAPIO_LL_DEBUG, "stonewalled %d",
                    ytc->ytc_num_ops_completed_before_stonewall);
            ytc->ytc_stonewalled = 1;
            break;
        }

        const yapio_blk_md_t *md = &md_array[j];

        if (!ytc->ytc_read)
            yapio_apply_contents_to_io_buffer(yapioIOBuf, ytg->ytg_blk_sz, md);

        off_t off = yapio_get_rw_offset(md, ytg->ytg_blk_sz);

        if (j % print_interval == 0)
            log_msg(YAPIO_LL_DEBUG, "rank=%d %s op=%d/%d off=%lu blk=%lu",
                    yapioMyRank, ytc->ytc_read ? "read" : "write",
                    j, ytc->ytc_num_ops_expected,
                    (unsigned long)off, md->ybm_blk_number);
        if (off < 0)
        {
            log_msg(YAPIO_LL_ERROR, "yapio_get_rw_offset() failed");
            rc = -ERANGE;
            break;
        }

        if (md->ybm_skip_io)
        {
            log_msg(YAPIO_LL_DEBUG, "skipping IO # %d, offset = %lu", j, off);
            continue;
        }

        ssize_t io_rc, io_bytes = 0;

        /* POSIX: loop to handle short reads/writes. */
        do
        {
            char *adjusted_buf    = yapioIOBuf + io_bytes;
            size_t adjusted_len   = ytg->ytg_blk_sz - io_bytes;
            off_t  adjusted_off   = off + io_bytes;

            int fd_idx = md->ybm_owner_rank_fpp;
            int fd = yapio_get_fd(fd_idx);

            io_rc = ytc->ytc_read ?
                YAPIO_SYS_CALL(pread)(fd, adjusted_buf, adjusted_len,
                                      adjusted_off) :
                YAPIO_SYS_CALL(pwrite)(fd, adjusted_buf, adjusted_len,
                                       adjusted_off);

            log_msg(YAPIO_LL_DEBUG, "%s rc=%zd off=%lu@%d fr=%d",
                    ytc->ytc_read ? "pread" : "pwrite", io_rc,
                    (unsigned long)adjusted_off, fd_idx,
                    ytg->ytg_first_rank);

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

        if (ytc->ytc_read && yapioVerifyRead)
        {
            rc = yapio_verify_contents_of_io_buffer(yapioIOBuf,
                                                    ytg->ytg_blk_sz, md);
            if (rc)
                break;
        }
    }

    if (!ytc->ytc_no_fsync && !yapioStoneWalled)
    {
        int fsync_rc = yapio_fsync();
        if (fsync_rc)
        {
            log_msg(YAPIO_LL_ERROR, "fsync(): %s", strerror(-rc));
            if (!rc)
                rc = fsync_rc;
        }
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
    const yapio_test_group_t *ytg = yapioMyTestGroup;
    bool fpp = ytg->ytg_file_per_process;

    const char *unlink_fn = fpp ? yapioTestFileNameFpp : yapioTestFileName;
    log_msg(YAPIO_LL_DEBUG, "%s keep=%d", unlink_fn, ytg->ytg_keep_file);

    if (ytg->ytg_keep_file || (!fpp && !yapio_leader_rank()))
        return;

    int rc = YAPIO_SYS_CALL(unlink)(unlink_fn);
    if (rc)
    {
        log_msg(YAPIO_LL_ERROR, "unlink %s: %s", unlink_fn, strerror(errno));

        yapio_exit(YAPIO_EXIT_ERR);
    }

    /* If the Metadata restart file is present then remove it too.
     */
    if (yapioRestartFileName[0] == '/')
        YAPIO_SYS_CALL(unlink)(yapioRestartFileName);
}

static void
yapio_test_ctx_release_md(yapio_test_ctx_md_t *ytcmh)
{
    if (ytcmh->ytcmh_ops)
    {
        YAPIO_FREE(ytcmh->ytcmh_ops);
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
    ytcmh->ytcmh_ops = YAPIO_CALLOC(nblks, sizeof(yapio_blk_md_t));

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

    int skips_left = ytc->ytc_sparse_io;
    int i;
    for (i = 0; i < num_ops; i++)
    {
        size_t src_idx = ytc->ytc_backwards ? num_ops - i - 1 : i;

        bool skip_io = skip_io_or_not(num_ops - i, skips_left);

        /* update writer in case of write (which isn't skipped) */
        if (!ytc->ytc_read && !skip_io)
        {
            log_msg(YAPIO_LL_DEBUG, "update writer rank for op #%d\n", i);
            yapio_source_md_update_writer_rank(src_idx, rank);
        }
#ifdef YAPIO_NIOVA
        /* In niova mode the vdev is persistent; a read run after a prior write
         * run will find real data at every block.  Mark blocks as not-hole so
         * yapio_verify_contents_of_io_buffer() actually verifies the content
         * instead of silently skipping it.
         */
        else if (ytc->ytc_read && !skip_io &&
                 yapioModeCurrent == YAPIO_IO_MODE_NIOVA)
        {
            yapioSourceBlkMd[src_idx].ybm_not_hole = 1;
        }
#endif

        md[i] = yapioSourceBlkMd[src_idx];

        if (skip_io)
        {
            log_msg(YAPIO_LL_DEBUG, "skip IO # %d\n", i);
            md[i].ybm_skip_io = 1;
            skips_left--;
        }

        unsigned long x = md[i].ybm_blk_number;
        log_msg(YAPIO_LL_TRACE, "writer_rank=%d md->ybm_blk_number=%zd",
                md[i].ybm_writer_rank, x);
    }
}

static int
yapio_blk_md_randomize(const yapio_blk_md_t *md_in, yapio_blk_md_t *md_out,
                       size_t num_mds, bool initialize_md_out)
{
    size_t buf_sz = num_mds * sizeof(int);
    int *array_of_randoms = YAPIO_MALLOC(buf_sz);
    if (!array_of_randoms)
        return -ENOMEM;

    int rc = yapio_read_from_dev_urandom((void *)array_of_randoms, buf_sz);
    if (rc)
    {
        YAPIO_FREE(array_of_randoms);
        return rc;
    }

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
        unsigned long swap_idx = array_of_randoms[i] % num_mds;
        yapio_blk_md_t md_tmp = md_out[swap_idx];
        md_out[swap_idx] = md_out[i];
        md_out[i] = md_tmp;

        unsigned long x = md_out[swap_idx].ybm_blk_number;
        unsigned long y = md_out[i].ybm_blk_number;

        log_msg(YAPIO_LL_DEBUG, "swapped %zu:%lu <-> %zu:%lu",
                i, x, swap_idx, y);
    }

    YAPIO_FREE(array_of_randoms);

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
    {
        rc = yapio_blk_md_randomize(yapioSourceBlkMd, md,
                                    ytg->ytg_num_blks_per_rank, true);

        select_holes(ytg->ytg_num_blks_per_rank, ytc->ytc_sparse_io, md);
    }

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
//    int skips_left = ytc->ytc_sparse_io;
    if (strided)
    {
        int i, j, total = 0;
        for (i = 0, total = 0; i < nranks; i++)
        {
            int skips_left = ytc->ytc_sparse_io;
            for (j = 0; j < nblks_div_nranks; j++, total++)
            {
                src_idx = (nranks * j) + i;
#if 0 //backwards + strided is broken
                if (ytc->ytc_backwards)
                    src_idx = (nranks * (nblks_div_nranks - j) - 1 + i);
#endif
                bool skip_io = skip_io_or_not(nblks_div_nranks - j, skips_left);

                if (!ytc->ytc_read && skip_io)
                {
                    log_msg(YAPIO_LL_DEBUG, "NN update writer rank for op total = %d\n", total);
                    yapio_source_md_update_writer_rank(src_idx, i);
                }

                md_send[total] = yapioSourceBlkMd[src_idx];
                if (skip_io)
                {
                    log_msg(YAPIO_LL_DEBUG, "skip IO # %d\n", i);
                    md_send[total].ybm_skip_io = 1;
                    skips_left--;
                }
            }
        }
    }
    else
    {
        yapio_blk_md_randomize(yapioSourceBlkMd, md_send, nblks_per_rank,
                               true);

        select_holes(ytg->ytg_num_blks_per_rank, ytc->ytc_sparse_io, md_send);

        if (!ytc->ytc_read)
        {
            for (src_idx = 0; src_idx < nblks_per_rank; src_idx++)
            {
                if (md_send[src_idx].ybm_skip_io)
                    continue;

                const int rank = src_idx / nblks_div_nranks;

                unsigned long send = md_send[src_idx].ybm_blk_number;
                unsigned long src = yapioSourceBlkMd[src_idx].ybm_blk_number;

                log_msg(YAPIO_LL_TRACE,
                        "%zu: blk-num:%zu:%zu old-rank=%d new-rank=%d",
                        src_idx, send, src, md_send[src_idx].ybm_writer_rank,
                        rank);

                /* This rank's yapioSourceBlkMd array must be updated with the
                 * rank which is about to write this block.
                 */
                const size_t update_idx =
                    md_send[src_idx].ybm_blk_number % nblks_per_rank;

                log_msg(YAPIO_LL_DEBUG, "update writer rank for op #%zu\n", src_idx);
                yapio_source_md_update_writer_rank(update_idx, rank);
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

        if (!ytc->ytc_read)
            ytg->ytg_last_writer_ctx = i;

        if (!i && ytc->ytc_read && !ytg->ytg_restart_from_previous_job
#ifdef YAPIO_NIOVA
            && yapioModeCurrent != YAPIO_IO_MODE_NIOVA
#endif
            )
        {
            if (yapio_leader_rank())
            {
                log_msg(YAPIO_LL_FATAL,
                        "Read test may not occur first unless restarting from"
                        " a previous job.");
            }
            else
            {
                yapio_exit(-EINVAL);
            }
        }

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

typedef struct yapio_per_rank_result
{
    bool          yprr_stonewalled;
    int           yprr_num_ops_completed;
    yapio_timer_t yprr_barrier_wait;
} yapio_per_rank_result_t;

static int
yapio_gather_barrier_stats_median_cmp(const void *a, const void *b)
{
    const yapio_per_rank_result_t *yppr_a = a;
    const yapio_per_rank_result_t *yppr_b = b;

    float a_val = yapio_timer_to_float(&yppr_a->yprr_barrier_wait);
    float b_val = yapio_timer_to_float(&yppr_b->yprr_barrier_wait);

    if (a_val > b_val)
        return 1;
    else if (a_val < b_val)
        return -1;

    return 0;
}

static void
yapio_gather_barrier_stats(yapio_test_ctx_t *ytc, bool leader_rank)
{
    const int nranks = ytc->ytc_group->ytg_num_ranks;

    float *barrier_global_results =
        leader_rank ? ytc->ytc_barrier_results : NULL;

    int *barrier_max_rank = leader_rank ? &ytc->ytc_barrier_max_rank : NULL;

    yapio_per_rank_result_t *all_results = NULL;

    yapio_per_rank_result_t yprr =
        {.yprr_stonewalled = ytc->ytc_stonewalled,
         .yprr_num_ops_completed = ytc->ytc_num_ops_completed_before_stonewall,
         .yprr_barrier_wait = ytc->ytc_barrier_wait[0]};

    if (barrier_global_results)
    {
        *barrier_max_rank = -1;
        barrier_global_results[YAPIO_BARRIER_STATS_MAX] = 0.0;
        barrier_global_results[YAPIO_BARRIER_STATS_AVG] = 0.0;

        /* This node is rank0 and will gather the timers for reporting.
         */
        all_results =
            YAPIO_CALLOC(nranks, sizeof(yapio_per_rank_result_t));

        if (!all_results)
            log_msg(YAPIO_LL_FATAL, "calloc: %s", strerror(ENOMEM));
    }

    MPI_OP_START;
    int rc = MPI_Gather(&yprr, sizeof(yapio_per_rank_result_t),
                        MPI_BYTE, all_results, sizeof(yapio_per_rank_result_t),
                        MPI_BYTE, 0, ytc->ytc_group->ytg_comm);
    MPI_OP_END;

    if (rc != MPI_SUCCESS)
        log_msg(YAPIO_LL_FATAL, "MPI_Gather: error=%d", rc);

    if (barrier_global_results)
    {
        ssize_t nblks_written_before_stonewalling = 0;
        int i;
        for (i = 0; i < nranks; i++)
        {
            float bwait =
                yapio_timer_to_float(&all_results[i].yprr_barrier_wait);

            barrier_global_results[YAPIO_BARRIER_STATS_AVG] += bwait;

            if (bwait > barrier_global_results[YAPIO_BARRIER_STATS_MAX])
            {
                barrier_global_results[YAPIO_BARRIER_STATS_MAX] = bwait;
                *barrier_max_rank = i;
            }

            /* Check if any ranks have been stonewalled.  If so, then adjust
             * the leaders ytc stats to reflect this.
             */
            if (all_results[i].yprr_stonewalled)
            {
                ytc->ytc_stonewalled = 1;
                nblks_written_before_stonewalling +=
                    all_results[i].yprr_num_ops_completed;
            }
        }

        /* Calculate the average number of blocks processed by each rank.
         */
        if (ytc->ytc_stonewalled)
            ytc->ytc_num_ops_completed_before_stonewall =
                nblks_written_before_stonewalling / nranks;

        log_msg(YAPIO_LL_DEBUG, "%d = ytc_num_ops_completed_before_stonewall",
                ytc->ytc_num_ops_completed_before_stonewall);

        /* Calculate average
         */
        barrier_global_results[YAPIO_BARRIER_STATS_AVG] /= nranks;

        /* Determine the median by sorting the result array.
         */
        qsort((void *)all_results, (size_t)nranks,
              sizeof(yapio_per_rank_result_t),
              yapio_gather_barrier_stats_median_cmp);

        barrier_global_results[YAPIO_BARRIER_STATS_MED] =
            yapio_timer_to_float(&all_results[nranks / 2].yprr_barrier_wait);

        YAPIO_FREE(all_results);
    }
}

static void
yapio_gather_test_barrier_results(yapio_test_ctx_t *ytc)
{
    yapio_gather_barrier_stats(ytc, yapio_leader_rank());
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
    if (yapio_leader_rank())
    {
        if (ytc->ytc_group->ytg_group_num > 0)
        {
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
    }

    MPI_OP_START;
    MPI_Bcast(&ytc->ytc_stonewalled, sizeof(bool), MPI_CHAR, 0,
              ytc->ytc_group->ytg_comm);
    MPI_OP_END;

    log_msg_r0(YAPIO_LL_DEBUG, "ytc_test_num=%d stonewalled=%d",
               ytc->ytc_test_num, ytc->ytc_stonewalled);
}

static void
yapio_display_result(const yapio_test_ctx_t *ytc, yapio_test_group_t *ytg)
{
    const yapio_timer_t *test_duration = &ytc->ytc_test_duration;
    const int nranks = ytg->ytg_num_ranks;
    const size_t blksz = ytg->ytg_blk_sz;
    const size_t nblks_per_rank =
        ytc->ytc_stonewalled ?
        (size_t)ytc->ytc_num_ops_completed_before_stonewall :
        ytg->ytg_num_blks_per_rank;

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

    fprintf(stdout, "%8.2f %s.%02d.%02d: %s%s%s%s%s%s%s %6.02f %siB/s%s",
            yapio_timer_to_float(&ytc->ytc_reported_time),
            ytc->ytc_suffix, ytg->ytg_group_num,
            ytc->ytc_test_num,
            (ytc->ytc_io_pattern == YAPIO_IOP_SEQUENTIAL ? "s" :
             (ytc->ytc_io_pattern ==
              YAPIO_IOP_RANDOM ? "R" : "S")),
            ytc->ytc_read            ? "r" : "w",
            ytc->ytc_remote_locality ? "D" : "L",
            ytc->ytc_no_fsync        ? "f" : "-",
            ytc->ytc_backwards       ? "b" : "-",
            ytc->ytc_leave_holes     ? "h" : "-",
            ytc->ytc_stonewalled     ? "X" : "-",
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

/* dump yapioMyRank's metadata array into a file */
static void
yapio_store_md_final_state(void)
{
    if (yapio_md_buffer_io(yapioMyTestGroup, true))
        log_msg(YAPIO_LL_FATAL, "yapio_md_buffer_io() failed");
}

static void
yapio_try_sync_md(const yapio_test_ctx_t *ytc, const bool last_writer_ctx)
{
    if (!ytc->ytc_read && ytc->ytc_group->ytg_keep_file)
    {
        if (ytc->ytc_stonewalled)
        {
            /* User has requested to keep the output data set but the
             * current write has been stonewalled.
             */
            ytc->ytc_group->ytg_keep_file = 0;
            log_msg(YAPIO_LL_WARN,
                    "Data set %s%s was stonewalled and will be removed.",
                    ytc->ytc_group->ytg_prefix, ytc->ytc_group->ytg_suffix);
        }
        else if (last_writer_ctx)
        {
            yapio_store_md_final_state();
        }
    }
}

static void
yapio_exec_all_tests(void)
{
    yapio_test_group_t *ytg = yapioMyTestGroup;

    yapio_mpi_barrier(yapioMyTestGroup->ytg_comm);

    log_msg_r0(YAPIO_LL_DEBUG,
               "g@%p nctxs=%d nranks=%d nblks=%zu blksz=%zu fpp=%d lr=%d %s",
               ytg, ytg->ytg_num_contexts, ytg->ytg_num_ranks,
               ytg->ytg_num_blks_per_rank, ytg->ytg_blk_sz,
               ytg->ytg_file_per_process, ytg->ytg_leader_rank,
               ytg->ytg_suffix);

    int i;
    for (i = 0; i < yapioMyTestGroup->ytg_num_contexts; i++)
    {
        yapio_test_ctx_t *ytc = &yapioMyTestGroup->ytg_contexts[i];
        ytc->ytc_group = yapioMyTestGroup;
        strncpy(ytc->ytc_suffix, yapioMyTestGroup->ytg_suffix,
                YAPIO_MKSTEMP_TEMPLATE_LEN + 1);
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

        yapioIOErrorCnt += yapio_perform_io(ytc);

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

        yapio_try_sync_md(ytc, i == ytg->ytg_last_writer_ctx ?
                          true : false);

        /* Free memory allocated in the test.
         */
        yapio_test_ctx_release(ytc);

        if (ytc->ytc_stonewalled)
        {
            log_msg(YAPIO_LL_WARN, "stonewalled, stopping");
            break;
        }
    }

    yapio_stat_ready();

    yapioHaltStonewallThread = true;
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

static int
yapio_stats_report_iterate(yapio_timer_t start_time)
{
    int remaining = 0;
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

                ytc->ytc_run_status = YAPIO_TEST_CTX_RUN_STATS_REPORTED;

                yapio_display_result(ytc, ytg);
            }

            if (ytc->ytc_run_status != YAPIO_TEST_CTX_RUN_STATS_REPORTED)
                remaining++;
        }
    }

    return remaining;
}

static void *
yapio_stats_reporting(void *unused_arg)
{
    int remaining_test_contexts_to_report = 0;

    yapio_timer_t start_time;
    yapio_get_time(&start_time);

    do
    {
        pthread_mutex_lock(&yapioThreadMutex);

        remaining_test_contexts_to_report =
            yapio_stats_report_iterate(start_time);

        if (remaining_test_contexts_to_report)
            pthread_cond_wait(&yapioThreadCond, &yapioThreadMutex);

        pthread_mutex_unlock(&yapioThreadMutex);
    } while (remaining_test_contexts_to_report && !yapioHaltStonewallThread);

    log_msg(YAPIO_LL_DEBUG,
            "remaining_test_contexts_to_report=%d yapioHaltStonewallThread=%d",
            remaining_test_contexts_to_report, yapioHaltStonewallThread);

    if (remaining_test_contexts_to_report)
        yapio_stats_report_iterate(start_time);

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
    while (remaining_reports && !yapioHaltStonewallThread)
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

static void *
yapio_stonewall_thread(void *unused_arg)
{
    if (yapioUseStoneWalling)
        do
        {
            if (!yapioStoneWallNsecs--)
            {
                yapioStoneWalled = true;
                log_msg(YAPIO_LL_DEBUG, "yapioStoneWalled=%d",
                        yapioStoneWalled);
                break;
            }
            else
            {
                sleep(1);
            }
        } while (!yapioHaltStonewallThread);

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

    if (yapioUseStoneWalling)
        pthread_join(yapioStoneWallThread, NULL);

    if (yapioNumTestGroups > 1)
        pthread_join(yapioStatsCollectionThread, NULL);
}

static void
yapio_stonewalling_thread_launch(void)
{
    if (pthread_create(&yapioStoneWallThread, NULL,
                       yapio_stonewall_thread, NULL))
        log_msg(YAPIO_LL_FATAL, "pthread_create: %s", strerror(errno));
}

static void
yapio_init_available_io_modes(void)
{
    yapio_io_modes[YAPIO_IO_MODE_DEFAULT] = &yapioDefaultSysCallOps;
    yapio_io_modes[YAPIO_IO_MODE_MMAP] = &yapioMmapSysCallOps;
#ifdef YAPIO_IME
    yapio_io_modes[YAPIO_IO_MODE_IME] = &yapioIMESysCallOps;
#endif
#ifdef YAPIO_MPIIO
    yapio_io_modes[YAPIO_IO_MODE_MPIIO] = &yapioMPIIOSysCallOps;
#endif
}

static void
yapio_io_mode_init(void)
{
#ifdef YAPIO_IME
    if (yapioModeCurrent == YAPIO_IO_MODE_IME)
        return ime_client_native2_init();
#endif
    return;
}

static int
yapio_io_mode_finalize(void)
{
#ifdef YAPIO_IME
    if (yapioModeCurrent == YAPIO_IO_MODE_IME)
        return ime_client_native2_finalize();
#endif

    return 0;
}

int
main(int argc, char *argv[])
{
    yapio_init_available_io_modes();

    yapio_mpi_setup(argc, argv);

    yapio_getopts(argc, argv);

    yapio_assign_rank_to_group();

    yapio_verify_test_contexts(yapioMyTestGroup);

    yapio_io_mode_init();

#ifdef YAPIO_NIOVA
    if (yapioModeCurrent == YAPIO_IO_MODE_NIOVA)
    {
        /* Connect to niova before allocating IO buffers. */
        yapio_niova_setup_clients();
    }
#endif

    yapio_setup_buffers(yapioMyTestGroup);

    if (yapioModeCurrent != YAPIO_IO_MODE_NIOVA)
        yapio_setup_test_file(yapioMyTestGroup);

    if (yapio_global_leader_rank())
        yapio_start_stats_collection_and_reporting_threads();

    if (yapioUseStoneWalling)
        yapio_stonewalling_thread_launch();

    yapio_mpi_barrier(MPI_COMM_WORLD);

    yapio_exec_all_tests();

    yapio_mpi_barrier(MPI_COMM_WORLD);

    if (yapioModeCurrent != YAPIO_IO_MODE_NIOVA)
        yapio_close_test_file(yapioMyTestGroup);

    if (yapio_global_leader_rank())
        yapio_destroy_collection_and_reporting_threads();

    yapio_destroy_buffers();

    if (yapioModeCurrent != YAPIO_IO_MODE_NIOVA)
        yapio_unlink_test_file();

#ifdef YAPIO_NIOVA
    if (yapioModeCurrent == YAPIO_IO_MODE_NIOVA)
        yapio_niova_teardown_clients();
#endif

    yapio_exit(yapio_io_mode_finalize());

    return 0;
}
