#!/bin/bash

TEST_DIR=/tmp/yapio.test.`date +%s`
CMD_NUM=0

mkdir ${TEST_DIR}
if [ $? -ne 0 ]
then
    echo "Failed to mkdir ${TEST_DIR}"
    exit 1
fi

run_cmd()
{
    eval "$@ > ${TEST_DIR}/log.${CMD_NUM}"
    rc=$?

    if [ $rc -ne 0 ]
    then
        echo "FAILED: $@ rc=$rc"
        exit $rc
    fi

    let CMD_NUM=$CMD_NUM+1

    echo "OK: $@"
}

# Ensure mpirun is available
run_cmd which mpirun

# Launch a few simple tests
run_cmd mpirun -np 1 ./yapio -t wsL,rsL ${TEST_DIR}
run_cmd mpirun -np 1 ./yapio -p "ssf." -s -k -t wsL,rsL ${TEST_DIR}
run_cmd mpirun -np 1 ./yapio -p "fpp." -s -k -t F:wsL,rsL ${TEST_DIR}
run_cmd mpirun -np 4 ./yapio -p "fpp-multi." -s -k -t F:wsL,rsL ${TEST_DIR}

run_cmd mpirun -np 8 ./yapio -S1 -t n12800:wsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,\
rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL /tmp/

run_cmd mpirun -np 8 ./yapio -S1 -t n12800:wsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,\
rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL -t wsL /tmp/

run_cmd mpirun -np 8 ./yapio -S10 -t n12800:wsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,\
rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL /tmp/

run_cmd mpirun -np 8 ./yapio -mm -s -S10 -t n128:wsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,\
rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL,rsL,rSD,rRD,rsL,rsL,rsL /tmp/

# Specify num blocks with -n
run_cmd mpirun -np 4 ./yapio -n 16 -t B1000:wSD,rsL,rRD,rRL,rSD \
    ${TEST_DIR}

# Specify num blocks inside cmd string
run_cmd mpirun -np 4 ./yapio -t n16:wsL,rsL,wRL,rRD,wRD,rsL,rsD ${TEST_DIR}

run_cmd mpirun -np 8 ./yapio -t N7:B8:n$((16*7)):wsL,rsL,wRL,rRD,wRD,rsL,rsD -t wsL ${TEST_DIR}

# Use mmap
run_cmd mpirun -np 8 ./yapio -mm -t F:B512:wsL,rsL,wRL,rRD,wRD,rsL,rsD \
    ${TEST_DIR}

run_cmd mpirun -np 8 ./yapio -mm -t B4096:wsL,rsL,wRL,rRD,wRD,rsL,rsD \
    ${TEST_DIR}

# Launch a compound test
run_cmd mpirun -np 8 ./yapio -t wsL,rsL,wRL,rRD,wRD,rsL,rsD \
    -t F:wsL,rsL,wRL,rRD,wRD,rsL,rsD ${TEST_DIR}

# Launch a compound test with different block sizes
run_cmd mpirun -np 8 ./yapio -k -t F:N4:B4096:wsL,rsL,wRL,rRD,wRD,rsL,rsD \
    -t N4:B512:wsL,rsL,wRL,rRD,wRD,rsL,rsD ${TEST_DIR}

# Prepare to read tests
let cmd=${CMD_NUM}-1

#cat ${TEST_DIR}/log.$cmd

SUFFIXa=`grep -m1 -e [A-Za-z0-9].00.00: ${TEST_DIR}/log.$cmd |  awk '{print $2}' | awk -F \. '{print $1}'`
SUFFIXb=`grep -m1 -e [A-Za-z0-9].01.00: ${TEST_DIR}/log.$cmd |  awk '{print $2}' | awk -F \. '{print $1}'`

run_cmd stat ${TEST_DIR}/.yapio.${SUFFIXa}.0.md
run_cmd stat ${TEST_DIR}/.yapio.${SUFFIXb}.0.md

run_cmd mpirun -np 4 ./yapio -i ${SUFFIXa} \
    -k -t F:N4:B4096:rsL,wsL,rsL ${TEST_DIR}

run_cmd mpirun -np 4 ./yapio -i ${SUFFIXb} \
    -t N4:B512:rsL,wsL,rsL ${TEST_DIR}

run_cmd md5sum ${TEST_DIR}/.yapio.${SUFFIXa}.0.md
MD5SUM=`md5sum ${TEST_DIR}/.yapio.${SUFFIXa}.0.md | awk '{print $1}'`

# Run this one a few more times
for i in {00..03}
do
    run_cmd mpirun -np 4 ./yapio -i ${SUFFIXa} \
        -k -t F:N4:B4096:rsL,wRD ${TEST_DIR}

    MD5SUM_NEW=`md5sum ${TEST_DIR}/.yapio.${SUFFIXa}.0.md | awk '{print $1}'`
    if [ $MD5SUM == $MD5SUM_NEW ]
    then
        echo "Metadata file contents of ${SUFFIXa} have not changed".
        exit 1;
    fi

    MD5SUM=MD5SUM_NEW
done

run_cmd mpirun -np 4 ./yapio -k -S1 -i ${SUFFIXa} -mm -t \
    F:N4:B4096:rsL,rRD,rSD ${TEST_DIR}

# Last run (with mmap) and cleanup
run_cmd mpirun -np 4 ./yapio -i ${SUFFIXa} -mm -t F:N4:B4096:rsL,rRD,rSD \
    ${TEST_DIR}

stat ${TEST_DIR}/.yapio.${SUFFIXa}.0.md 2>/dev/null >/dev/null
if [ $? -eq 0 ]
then
    echo "Failed to remove md file: ${TEST_DIR}/.yapio.${SUFFIXa}.0.md"
    exit 1;
fi

stat ${TEST_DIR}/.yapio.${SUFFIXb}.0.md 2>/dev/null >/dev/null
if [ $? -eq 0 ]
then
    echo "Failed to remove md file: ${TEST_DIR}/.yapio.${SUFFIXb}.0.md"
    exit 1;
fi

#ls -lrt ${TEST_DIR}

# Remove Test Dir
run_cmd rm -fr ${TEST_DIR}
