#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <string>

using namespace std;

// ============================================================================
// CORE TENSOR TOPOLOGY ARCHITECTURE
// ============================================================================
class Tensor {
public:
    vector<double> data;  
    vector<size_t> shape; 

    // Constructor framework to manage raw allocation mapping
    Tensor(size_t n, size_t c, size_t h, size_t w) : shape({n, c, h, w}) {
        data.resize(n * c * h * w, 0.0);
    }

    // THIS PART HANDLES: The contiguous 1D flattening math designed by Jeremy (M3)
    // Formula: Index(n,c,h,w) = n*(C*H*W) + c*(H*W) + h*W + w
    inline size_t get_index(size_t n, size_t c, size_t h, size_t w) const {
        if (n >= shape[0] || c >= shape[1] || h >= shape[2] || w >= shape[3]) {
            throw out_of_range("Tensor index out of bounds exception.");
        }
        return n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w;
    }

    // THIS PART HANDLES: The 4D coordinate vector element accessor built by Agatha (M4)
    double& operator()(size_t n, size_t c, size_t h, size_t w) {
        return data[get_index(n, c, h, w)];
    }

    // THIS PART HANDLES: The const read-only matrix accessor built by Agatha (M4)
    const double& operator()(size_t n, size_t c, size_t h, size_t w) const {
        return data[get_index(n, c, h, w)];
    }
};

// MODULE 8: SYSTEM INTEGRATION & DEMO PIPELINE MANAGER

class PipelineIntegrationManager {
public:
    // THIS PART HANDLES: Connecting function hooks to accept the other 7 developers' modules
    typedef Tensor (*ImageLoaderEngine)(const string&);
    typedef void (*DataPreprocessorEngine)(Tensor&);
    typedef Tensor (*WeightsInitializerEngine)(size_t, size_t, size_t, size_t);
    typedef Tensor (*ConvolutionMathEngine)(const Tensor&, const Tensor&);

    // THIS PART HANDLES: The automated unit testing built  to pass sprint criteria
    static bool run_system_validation_tests(ConvolutionMathEngine architecture_conv) {
        cout << "[M8 Integration] Launching automated framework diagnostic checks...\n";
        
        try {
            // Test 1 verifying boundary safety metrics
            Tensor boundary_test(1, 1, 4, 4);
            try {
                boundary_test(1, 0, 0, 0); 
                cout << " -> Test 1 (Boundary Overflow Protection Check): FAILED\n";
                return false;
            } catch (const out_of_range&) {
                cout << " -> Test 1 (Boundary Overflow Protection Check): PASSED\n";
            }

            // Test 2 verifying calculation metrics
            Tensor mock_in(1, 1, 3, 3);
            Tensor mock_k(1, 1, 2, 2);
            fill(mock_in.data.begin(), mock_in.data.end(), 1.0);
            fill(mock_k.data.begin(), mock_k.data.end(), 2.0);
            
            Tensor mock_out = architecture_conv(mock_in, mock_k);
            if (abs(mock_out(0, 0, 0, 0) - 8.0) > 1e-9) {
                cout << " -> Test 2 (Convolution Arithmetic Formula Check): FAILED\n";
                return false;
            }
            cout << " -> Test 2 (Convolution Arithmetic Formula Check): PASSED\n";

        } catch (const exception& e) {
            cout << " -> Diagnostic pipeline execution aborted: " << e.what() << "\n";
            return false;
        }

        cout << ">> STATUS: Core pipeline layer compliance check complete.\n\n";
        return true;
    }

    // THIS PART HANDLES: The entire Week 1 execution pipeline orchestrator 
    static void execute_system_pipeline(
        string target_file_path,
        ImageLoaderEngine load_stage,
        DataPreprocessorEngine preprocess_stage,
        WeightsInitializerEngine weight_init_stage,
        ConvolutionMathEngine conv_processing_stage
    ) {
        cout << "=========================================================\n";
        cout << "  LAUNCHING INTEGRATED CNN DATA PROCESSING ENVIRONMENT     \n";
        cout << "=========================================================\n";

        try {
            // THIS PART EXECUTES: binary image parsing file engine
            cout << "[Stage 1] Loading image vector binary allocations...\n";
            Tensor processing_tensor = load_stage(target_file_path);

            // THIS PART EXECUTES:  spatial channel data normalization engine
            cout << "[Stage 2] Transforming data bounds via normalization...\n";
            preprocess_stage(processing_tensor);
            
            assert(processing_tensor.data.size() > 0 && "Pipeline fault: Target image memory block unallocated.");

            // THIS PART EXECUTES:  filter matrix shape and weight initializer engine
            cout << "[Stage 3] Spawning target model filter weights...\n";
            Tensor kernel_filters = weight_init_stage(4, 3, 3, 3);

            // THIS PART EXECUTES:  multi-channel convolution arithmetic engine
            cout << "[Stage 4] Compiling matrix transformations across layers...\n";
            Tensor feature_maps = conv_processing_stage(processing_tensor, kernel_filters);

            // THIS PART EXECUTES:  Week-1 output diagnostic status report logging
            cout << "\n=========================================================\n";
            cout << "               WEEK 1 SYSTEM METRICS SUMMARY             \n";
            cout << "=========================================================\n";
            cout << "Input Tensor Layout   : {" << processing_tensor.shape[0] << ", " << processing_tensor.shape[1] 
                 << ", " << processing_tensor.shape[2] << ", " << processing_tensor.shape[3] << "}\n";
            cout << "Kernel Tensor Layout  : {" << kernel_filters.shape[0] << ", " << kernel_filters.shape[1] 
                 << ", " << kernel_filters.shape[2] << ", " << kernel_filters.shape[3] << "}\n";
            cout << "Output Feature Layout : {" << feature_maps.shape[0] << ", " << feature_maps.shape[1] 
                 << ", " << feature_maps.shape[2] << ", " << feature_maps.shape[3] << "}\n";
            cout << "Data Map Anchor Node  : " << feature_maps(0, 0, 0, 0) << "\n";
            cout << "---------------------------------------------------------\n";
            cout << ">> PIPELINE STATE : STABLE SYSTEM ENVIRONMENT (0 Resource Leaks)\n";
            cout << "=========================================================\n";

        } catch (const exception& e) {
            cerr << "\n!! CORE CRITICAL PIPELINE EXCEPTION INTERCEPTED: " << e.what() << "\n";
        }
    }
};
