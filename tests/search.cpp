#include <thread>
#include <omp.h>
#include <chrono>
#include "../hnswlib/hnswlib.h"

void save_result(char* filename, std::vector<std::vector<unsigned>>& results) {
    std::ofstream out(filename, std::ios::binary | std::ios::out);

    for (unsigned i = 0; i < results.size(); i++) {
        unsigned GK = (unsigned)results[i].size();
        out.write((char*)&GK, sizeof(unsigned));
        out.write((char*)results[i].data(), GK * sizeof(unsigned));
    }
    out.close();
}

template <class T>
T* read_bin(const char* filename, unsigned& npts, unsigned& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(&npts), sizeof(unsigned));
    in.read(reinterpret_cast<char*>(&dim), sizeof(unsigned));
    std::cout << "Loading data from file: " << filename << ", points_num: " << npts << ", dim: " << dim << std::endl;
    size_t total_size = static_cast<size_t>(npts) * dim;  // notice if the space is exceed the bound size_t max
    if (total_size > std::numeric_limits<size_t>::max() / sizeof(T)) {
        std::cerr << "Requested size is too large." << std::endl;
        exit(EXIT_FAILURE);
    }
    T* data = new T[total_size];
    in.read(reinterpret_cast<char*>(data), total_size * sizeof(T));
    in.close();
    return data;
}

float ComputeRecall(unsigned q_num, unsigned k, unsigned gt_dim, std::vector<std::vector<unsigned>> res, unsigned* gt) {
    unsigned total_count = 0;
    for (unsigned i = 0; i < q_num; i++) {
        std::vector<unsigned> one_gt(gt + i * gt_dim, gt + i * gt_dim + k);
        std::vector<unsigned> intersection;
        std::vector<unsigned> temp_res = res[i];
        for (auto p : one_gt) {
            if (std::find(temp_res.begin(), temp_res.end(), p) != temp_res.end())
                intersection.push_back(p);
        }

        total_count += static_cast<unsigned>(intersection.size());
    }
    return static_cast<float>(total_count) / (float)(k * q_num);
}

struct SearchStats {
    std::vector<std::pair<float, hnswlib::labeltype>> results;
    unsigned computations;  // 距离计算次数
    unsigned hops;         // 访问的节点数量
};

// 扩展HierarchicalNSW类来支持统计信息
template<typename dist_t>
class CustomHNSW : public hnswlib::HierarchicalNSW<dist_t> {
public:
    using hnswlib::HierarchicalNSW<dist_t>::HierarchicalNSW;

    // 添加新的搜索函数，返回统计信息
    SearchStats searchKnnWithStats(const void* query_data, size_t k) {
        SearchStats stats;
        stats.computations = 0;
        stats.hops = 0;

        this->metric_distance_computations = 0;
        this->metric_hops = 0;
        auto top_candidates = this->searchKnn(query_data, k);

        while (!top_candidates.empty()) {
            stats.results.push_back(top_candidates.top());
            top_candidates.pop();
        }

        stats.computations = this->metric_distance_computations;
        stats.hops = this->metric_hops;

        return stats;
    }
};

int main(int argc, char** argv) {
    if (argc != 8) {
        std::cout << argv[0]
                  << " data_file query_file gt_file hnsw_path search_K threads result_path"
                  << std::endl;
        exit(-1);
    }
    // float* data_load = NULL;
    unsigned points_num, dim;
    // load_data(argv[1], data_load, points_num, dim);
    float* data_load = read_bin<float>(argv[1], points_num, dim);

    // float* query_load = NULL;
    unsigned query_num, query_dim;
    // load_data(argv[2], query_load, query_num, query_dim);
    float* query_load = read_bin<float>(argv[2], query_num, query_dim);

    unsigned gt_num, gt_dim;
    unsigned* gt_load = read_bin<unsigned>(argv[3], gt_num, gt_dim);

    assert(dim == query_dim);
    assert(query_num == gt_num);

    std::vector<unsigned> Ls = {10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
                                100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 220, 240, 260, 280, 300, 350, 400, 450, 500, 550, 600, 650,
                                700, 750, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 4000
                                //4500,5000,6000,8000,10000
                                };

    unsigned K = (unsigned)atoi(argv[5]);
    unsigned T = (unsigned)atoi(argv[6]);
    omp_set_num_threads(T);
    std::cout << "using threads: " << T << "   topk: " << K << std::endl;

    const char* result_path = argv[7];
    std::ofstream evaluation_out;
    evaluation_out.open(result_path, std::ios::out); //  | std::ios::app
    if (evaluation_out.is_open()) {
        evaluation_out << "L_pq,QPS,avg_visited,mean_latency,recall@" << K << ",avg_hops" << std::endl;
    } else {
        std::cerr << "Error! Result path is wrong!" << std::endl;
    }

    hnswlib::InnerProductSpace space(dim);
    std::unique_ptr<CustomHNSW<float>> index;
    try {
        index.reset(new CustomHNSW<float>(&space, argv[4]));
    } catch (const std::exception& e) {
        std::cerr << "Error loading index: " << e.what() << std::endl;
        exit(-1);
    }

    std::cout << "start search." << std::endl;
    std::cout << "L_pq" << "\t\tQPS" << "\t\t\tavg_visited" << "\tmean_latency" << "\trecall@" << K << "\tavg_hops" << std::endl;

    for (unsigned L : Ls) {
        if (L < K) continue;

        index->setEf(L);

        std::vector<std::vector<unsigned>> res(query_num);
        std::vector<unsigned> projection_cmps_vec(query_num);
        std::vector<unsigned> hops_vec(query_num);

#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < query_num; i++) {
            auto stats = index->searchKnnWithStats(query_load + i * dim, K);
            std::vector<unsigned> tmp;
            for (const auto& result : stats.results) {
                tmp.push_back(result.second);
            }
            res[i] = tmp;
            // 保存统计信息
            projection_cmps_vec[i] = stats.computations;
            hops_vec[i] = stats.hops;
        }

        auto s = std::chrono::high_resolution_clock::now();
#pragma omp parallel for schedule(dynamic, 1)
        for (unsigned i = 0; i < query_num; i++) {
            index->searchKnn(query_load + i * dim, K);
        }
        auto e = std::chrono::high_resolution_clock::now();

        auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(e - s).count();
        float qps = (float)query_num / ((float)diff / 1000.0);
        float recall = ComputeRecall(query_num, K, gt_dim, res, gt_load);

        float avg_projection_cmps = 0.0;
        float avg_hops = 0.0;
        for (size_t i = 0; i < query_num; ++i) {
            avg_projection_cmps += projection_cmps_vec[i];
            avg_hops += hops_vec[i];
        }
        avg_projection_cmps /= query_num;
        avg_hops /= (float)query_num;

        std::cout << L << "\t\t" << qps << "\t\t" << avg_projection_cmps << "\t\t"
                  << ((float)diff / query_num) << "\t\t" << recall << "\t\t" << avg_hops << std::endl;
        if (evaluation_out.is_open()) {
            evaluation_out << L << "," << qps << "," << avg_projection_cmps << "," << ((float)diff / query_num) << ","
                           << recall << "," << avg_hops << std::endl;
        }
    }

    if (evaluation_out.is_open()) {
        evaluation_out.close();
    }

    std::cout << "save result to: " << result_path << std::endl;

    delete[] data_load;
    delete[] query_load;
    delete[] gt_load;

    return 0;
}