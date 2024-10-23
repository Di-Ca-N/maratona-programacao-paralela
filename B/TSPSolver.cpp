#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <limits>
#include <fstream>
#include <sstream>
#include <string>
#include <omp.h>

struct Tour {
    std::vector<int> tour;
    double length;
};

class TSPSolver {
public:
    TSPSolver(unsigned seed)
        : gen_(seed) {
        loadAdjacencyMatrix();
    }

    std::vector<int> solveTSP(int numIterations, int numRestarts) {
        return shotgunHillClimbing(numIterations, numRestarts);
    }

    double calculateTourLength(const std::vector<int>& tour) const {
        double length = 0.0;
        //#pragma omp parallel for reduction(+:length) schedule(static, 4)
        for (size_t i = 0; i < tour.size(); ++i) {
            length += adjacencyMatrix_[tour[i]][tour[(i + 1) % tour.size()]];
        }
        return length;
    }

private:
    std::vector<std::vector<double>> adjacencyMatrix_;
    std::mt19937 gen_;

    void loadAdjacencyMatrix() {
        std::string line;
        while (std::getline(std::cin, line)) {
            adjacencyMatrix_.push_back(parseCSVLine(line));
        }

        if (adjacencyMatrix_.empty() || !isSquareMatrix()) {
            throw std::runtime_error("Invalid adjacency matrix in CSV file");
        }
    }

    std::vector<double> parseCSVLine(const std::string& line) {
        std::vector<double> row;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) {
            row.push_back(std::stod(cell));
        }
        return row;
    }

    bool isSquareMatrix() const {
        size_t size = adjacencyMatrix_.size();
        return std::all_of(adjacencyMatrix_.begin(), adjacencyMatrix_.end(),
                           [size](const auto& row) { return row.size() == size; });
    }

    std::vector<int> generateRandomTour() {
        std::vector<int> tour(adjacencyMatrix_.size());
        std::iota(tour.begin(), tour.end(), 0);
        #pragma omp critical (tourShuffle)
        {
            std::shuffle(tour.begin() + 1, tour.end(), gen_);
        }
        return tour;
    }

    Tour twoOptSwap(const std::vector<int>& tour, int i, int j, double currentLength) {
        std::vector<int> newTour = tour;
        std::reverse(newTour.begin() + i, newTour.begin() + j + 1);
        double newLength = currentLength - adjacencyMatrix_[tour[i - 1]][tour[i]] - adjacencyMatrix_[tour[j]][tour[(j + 1) % tour.size()]] + adjacencyMatrix_[tour[i - 1]][tour[j]] + adjacencyMatrix_[tour[i]][tour[(j + 1) % tour.size()]];
        return {newTour, newLength};
    }

    std::vector<int> shotgunHillClimbing(int numIterations, int numRestarts) {
        std::pair<std::vector<int>, double> result = {std::vector<int>(), std::numeric_limits<double>::max()};

        #pragma omp declare reduction(mintour : std::pair<std::vector<int>, double> : omp_out = (omp_in.second < omp_out.second) ? omp_in : omp_out) initializer(omp_priv = omp_orig)
        //int chunkSize = numRestarts / omp_get_max_threads();
        //printf("Restarts: %d, Threads: %d, chunks: %d\n", numRestarts, omp_get_max_threads(), chunkSize);
        #pragma omp parallel for reduction(mintour:result) schedule(static, 8)
        for (int restart = 0; restart < numRestarts; ++restart) {
            auto runValue = hillClimb(numIterations);
            if (runValue.second < result.second) {
                result = runValue;
            }
        }

        return result.first;
    }

    std::pair<std::vector<int>, double> hillClimb(int numIterations) {
        std::vector<int> currentTour = generateRandomTour();
        double currentLength = calculateTourLength(currentTour);

        for (int iter = 0; iter < numIterations; ++iter) {
            bool improvement = false;
            for (size_t i = 1; i < currentTour.size() - 1; ++i) {
                for (size_t j = i + 1; j < currentTour.size(); ++j) {
                    Tour a = twoOptSwap(currentTour, i, j, currentLength);
                    std::vector<int> newTour = a.tour;
                    double newLength = a.length;
                    if (newLength < currentLength) {
                        currentTour = std::move(newTour);
                        currentLength = newLength;
                        improvement = true;
                        break;
                    }
                }
                if (improvement) break;
            }
            if (!improvement) break;
        }

        return {currentTour, currentLength};
    }
};

int main(int argc, char* argv[]) {
    try {
        std::string line;
        std::getline(std::cin, line);
        std::stringstream myStream(line);
        std::getline(myStream, line, ' ');
        int numIterations = std::stoi(line);
        std::getline(myStream, line, ' ');
        int numRestarts = std::stoi(line);
        std::getline(myStream, line, ' ');
        unsigned seed = std::stoi(line);

        TSPSolver solver(seed);
        std::vector<int> bestTour = solver.solveTSP(numIterations, numRestarts);

        std::cout << "Best tour found: ";
        for (int vertex : bestTour) {
            std::cout << vertex << " ";
        }
        std::cout << "\nTour length: " << solver.calculateTourLength(bestTour) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
