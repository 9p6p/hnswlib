
dataset=t2i-10M
prefix=/root/datasets/${dataset}
save_prefix=/root/indices/${dataset}
results=/root/results/${dataset}

cd build

./search ${prefix}/base.10M.fbin \
 ${prefix}/query.10k.fbin \
 ${prefix}/gt.10k.ibin \
 ${save_prefix}/t2i_10M_hnswip.index \
 1 \
 1 \
 ${results}/test_search_t2i_10M_hnswip_top1_T1.csv