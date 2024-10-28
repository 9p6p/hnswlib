#include "../hnswlib/hnswlib.h"
#include <thread>
#include <omp.h>

// Multithreaded executor
// The helper function copied from python_bindings/bindings.cpp (and that itself is copied from nmslib)
// An alternative is using #pragme omp parallel for or any other C++ threading
template<class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads, Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value that
                         * size_t can fit, because fetch_add returns the previous value
                         * before the increment (what will result in overflow
                         * and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

template <class T>
T* read_bin(const char* filename, int& npts, int& dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    in.read(reinterpret_cast<char*>(&npts), sizeof(int));
    in.read(reinterpret_cast<char*>(&dim), sizeof(int));
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

int main() {
    
    int dim;               // Dimension of the elements
    int max_elements;   // Maximum number of elements, should be known beforehand
    float* data = read_bin<float>("/root/datasets/t2i-10M/base.10M.fbin", max_elements, dim);

    int M = 32;                 // Tightly connected with internal dimensionality of the data
                                // strongly affects the memory consumption
    int ef_construction = 500;  // Controls index search speed/build speed tradeoff
    int num_threads = omp_get_num_procs();       // Number of threads for operations with index

    // Initing index
    hnswlib::InnerProductSpace space(dim);
    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, max_elements, M, ef_construction);
    auto s = std::chrono::high_resolution_clock::now();
    // Add data to index
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        alg_hnsw->addPoint((void*)(data + dim * row), row);
    });

    auto e = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = e - s;
    std::cout << "indexing time: " << diff.count() << "\n";
    alg_hnsw->saveIndex("/root/indices/t2i-10M/hnsw.l2test");
    
    // Query the elements for themselves and measure recall
    std::vector<hnswlib::labeltype> neighbors(max_elements);
    ParallelFor(0, max_elements, num_threads, [&](size_t row, size_t threadId) {
        std::priority_queue<std::pair<float, hnswlib::labeltype>> result = alg_hnsw->searchKnn(data + dim * row, 1);
        hnswlib::labeltype label = result.top().second;
        neighbors[row] = label;
    });
    float correct = 0;
    for (int i = 0; i < max_elements; i++) {
        hnswlib::labeltype label = neighbors[i];
        if (label == i) correct++;
    }
    float recall = correct / max_elements;
    std::cout << "Recall: " << recall << "\n";

    delete[] data;
    delete alg_hnsw;
    return 0;
}