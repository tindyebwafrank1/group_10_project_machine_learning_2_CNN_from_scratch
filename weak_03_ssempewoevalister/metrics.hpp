#ifndef METRICS_HPP
#define METRICS_HPP

// Week 3 - Evalister: Evaluation metrics suite.
// Header-only, namespace CNN, so it builds with the single-command release:
//   g++ -O3 -Wall -Wextra -std=c++17 main.cpp -o cnn_app

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace CNN {

    // Rows = actual class, columns = predicted class.
    class ConfusionMatrix {
    private:
        size_t num_classes;
        std::vector<size_t> counts; // flattened [actual * num_classes + predicted]
        size_t total_samples;

        void check_class(size_t c) const {
            if (c >= num_classes)
                throw std::out_of_range("Class index " + std::to_string(c) +
                                        " out of range [0, " + std::to_string(num_classes) + ")");
        }

    public:
        explicit ConfusionMatrix(size_t classes)
            : num_classes(classes), counts(classes * classes, 0), total_samples(0) {
            if (classes == 0)
                throw std::invalid_argument("ConfusionMatrix needs at least 1 class");
        }

        size_t classes() const { return num_classes; }
        size_t total() const { return total_samples; }

        // Record one prediction.
        void add(size_t actual, size_t predicted) {
            check_class(actual);
            check_class(predicted);
            ++counts[actual * num_classes + predicted];
            ++total_samples;
        }

        // Record many predictions (e.g. one mini-batch from the argmax module).
        void add_batch(const std::vector<size_t>& actual, const std::vector<size_t>& predicted) {
            if (actual.size() != predicted.size())
                throw std::invalid_argument("actual and predicted must have the same length");
            for (size_t i = 0; i < actual.size(); ++i) {
                check_class(actual[i]);
                check_class(predicted[i]);
            }
            // Validated first, so a bad batch never leaves the matrix half-updated.
            for (size_t i = 0; i < actual.size(); ++i) add(actual[i], predicted[i]);
        }

        void reset() {
            counts.assign(num_classes * num_classes, 0);
            total_samples = 0;
        }

        size_t count(size_t actual, size_t predicted) const {
            check_class(actual);
            check_class(predicted);
            return counts[actual * num_classes + predicted];
        }

        // Per-class (one-vs-rest) counts.
        size_t true_positives(size_t c) const { return count(c, c); }

        size_t false_positives(size_t c) const { // predicted c, actually other
            check_class(c);
            size_t s = 0;
            for (size_t a = 0; a < num_classes; ++a)
                if (a != c) s += counts[a * num_classes + c];
            return s;
        }

        size_t false_negatives(size_t c) const { // actually c, predicted other
            check_class(c);
            size_t s = 0;
            for (size_t p = 0; p < num_classes; ++p)
                if (p != c) s += counts[c * num_classes + p];
            return s;
        }

        size_t true_negatives(size_t c) const {
            return total_samples - true_positives(c) - false_positives(c) - false_negatives(c);
        }

        // Number of samples whose actual class is c.
        size_t support(size_t c) const { return true_positives(c) + false_negatives(c); }

        // ---- Metrics. Any 0/0 case returns 0.0 (documented in docs/metrics.md). ----

        double accuracy() const {
            if (total_samples == 0) return 0.0;
            size_t correct = 0;
            for (size_t c = 0; c < num_classes; ++c) correct += counts[c * num_classes + c];
            return static_cast<double>(correct) / static_cast<double>(total_samples);
        }

        double precision(size_t c) const {
            size_t tp = true_positives(c), fp = false_positives(c);
            return (tp + fp == 0) ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fp);
        }

        double recall(size_t c) const {
            size_t tp = true_positives(c), fn = false_negatives(c);
            return (tp + fn == 0) ? 0.0 : static_cast<double>(tp) / static_cast<double>(tp + fn);
        }

        double f1(size_t c) const {
            double p = precision(c), r = recall(c);
            return (p + r == 0.0) ? 0.0 : 2.0 * p * r / (p + r);
        }

        // Macro average: unweighted mean over all classes.
        double macro_precision() const { return macro(&ConfusionMatrix::precision); }
        double macro_recall() const { return macro(&ConfusionMatrix::recall); }
        double macro_f1() const { return macro(&ConfusionMatrix::f1); }

        // Pretty-printed matrix plus per-class report, for logs and demos.
        std::string to_string() const {
            std::ostringstream os;
            os << "Confusion Matrix (rows = actual, cols = predicted)\n      ";
            for (size_t p = 0; p < num_classes; ++p) os << std::setw(6) << ("P" + std::to_string(p));
            os << "\n";
            for (size_t a = 0; a < num_classes; ++a) {
                os << std::setw(5) << ("A" + std::to_string(a)) << " ";
                for (size_t p = 0; p < num_classes; ++p) os << std::setw(6) << counts[a * num_classes + p];
                os << "\n";
            }
            os << std::fixed << std::setprecision(4);
            os << "\nClass  Precision  Recall  F1      Support\n";
            for (size_t c = 0; c < num_classes; ++c)
                os << std::setw(5) << c << "  " << std::setw(9) << precision(c) << "  "
                   << std::setw(6) << recall(c) << "  " << std::setw(6) << f1(c) << "  "
                   << std::setw(7) << support(c) << "\n";
            os << "\nAccuracy:        " << accuracy()
               << "\nMacro Precision: " << macro_precision()
               << "\nMacro Recall:    " << macro_recall()
               << "\nMacro F1:        " << macro_f1() << "\n";
            return os.str();
        }

    private:
        double macro(double (ConfusionMatrix::*fn)(size_t) const) const {
            double sum = 0.0;
            for (size_t c = 0; c < num_classes; ++c) sum += (this->*fn)(c);
            return sum / static_cast<double>(num_classes);
        }
    };

    // Convenience: build a matrix straight from label vectors.
    inline ConfusionMatrix build_confusion_matrix(const std::vector<size_t>& actual,
                                                  const std::vector<size_t>& predicted,
                                                  size_t num_classes) {
        ConfusionMatrix cm(num_classes);
        cm.add_batch(actual, predicted);
        return cm;
    }

    // Convenience: accuracy directly from label vectors (no class count needed).
    inline double accuracy(const std::vector<size_t>& actual, const std::vector<size_t>& predicted) {
        if (actual.size() != predicted.size())
            throw std::invalid_argument("actual and predicted must have the same length");
        if (actual.empty()) return 0.0;
        size_t correct = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i] == predicted[i]) ++correct;
        return static_cast<double>(correct) / static_cast<double>(actual.size());
    }
}

#endif // METRICS_HPP
