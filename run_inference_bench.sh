#
# jemalloc: https://github.com/jemalloc/jemalloc/wiki/Getting-Started
   export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:-1,muzzy_decay_ms:-1";
   export LD_PRELOAD=/home/mingfeim/packages/jemalloc-5.2.1/lib/libjemalloc.so

INPUT_FILE="RWKV_in_150_lines.py"

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`
LAST_CORE=`expr $CORES - 1`

KMP_BLOCKTIME=1

PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"

export KMP_BLOCKTIME=$KMP_BLOCKTIME

echo -e "### using KMP_BLOCKTIME=$KMP_BLOCKTIME\n"

## single socket test
echo -e "\n### using OMP_NUM_THREADS=$CORES"
PREFIX="numactl --physcpubind=0-$LAST_CORE --membind=0"
echo -e "### using $PREFIX\n"
OMP_NUM_THREADS=$CORES $PREFIX python -u $INPUT_FILE
