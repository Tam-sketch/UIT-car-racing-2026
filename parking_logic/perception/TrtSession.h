// ============================================================
// perception/TrtSession.h
//
// Generic TensorRT engine session.
//   - Loads a serialized .engine file
//   - Allocates device buffers for all bindings
//   - Runs one inference pass given host-side buffers
//
// NOTHING model-specific here. No BGR/RGB, no normalization,
// no channel-planarization, no argmax, no NMS. That belongs in
// the model wrapper (RoadSegmenter, SignDetector, ...).
//
// Assumes static shapes (no dynamic bindings).
// I/O may be FP32 or FP16 (auto-detected per binding).
// Assumes NCHW layouts. Multiple inputs/outputs supported via
// Run(inputs, outputs), but most callers use RunSingle().
//
// NOTE: uses TensorRT 8.x APIs (getBindingDataType, enqueueV2).
//       TRT 10+ removed these — a port will be needed if the
//       JetPack image is upgraded.
// ============================================================
#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cuda_fp16.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ucr {
namespace perception {

class TrtSession {
public:
    struct Binding {
        std::string         name;
        bool                isInput;
        nvinfer1::Dims      dims;
        nvinfer1::DataType  dataType{ nvinfer1::DataType::kFLOAT };
        size_t              elemCount;              // product of dims
        size_t              elemSize{ sizeof(float) };  // 2 for FP16, 4 for FP32
        size_t              byteSize;               // elemCount * elemSize
    };

    TrtSession() = default;
    ~TrtSession() { Shutdown(); }
    TrtSession(const TrtSession&)            = delete;
    TrtSession& operator=(const TrtSession&) = delete;

    // --------------------------------------------------------
    // Lifecycle
    // --------------------------------------------------------
    bool Init(const std::string& enginePath) {
        if (IsReady()) Shutdown();
        try {
            // Read file
            std::ifstream f(enginePath, std::ios::binary);
            if (!f) throw std::runtime_error("cannot open engine: " + enginePath);
            f.seekg(0, std::ios::end);
            size_t size = static_cast<size_t>(f.tellg());
            f.seekg(0, std::ios::beg);
            std::vector<char> blob(size);
            f.read(blob.data(), size);

            // Deserialize
            runtime_ = nvinfer1::createInferRuntime(logger_);
            if (!runtime_) throw std::runtime_error("createInferRuntime failed");

            engine_ = runtime_->deserializeCudaEngine(blob.data(), blob.size());
            if (!engine_) throw std::runtime_error("deserializeCudaEngine failed");

            context_ = engine_->createExecutionContext();
            if (!context_) throw std::runtime_error("createExecutionContext failed");

            if (!BuildBindings()) throw std::runtime_error("BuildBindings failed");
            if (!AllocBuffers())  throw std::runtime_error("AllocBuffers failed");

            cudaError_t err = cudaStreamCreate(&stream_);
            if (err != cudaSuccess)
                throw std::runtime_error(std::string("cudaStreamCreate: ") +
                                         cudaGetErrorString(err));
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[TrtSession] Init error: " << e.what() << "\n";
            Shutdown();
            return false;
        }
    }

    void Shutdown() {
        if (stream_) {
            cudaStreamSynchronize(stream_);
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        for (void*& b : deviceBuffers_) {
            if (b) { cudaFree(b); b = nullptr; }
        }
        deviceBuffers_.clear();
        bindings_.clear();
        inputIdxs_.clear();
        outputIdxs_.clear();
        if (context_) { delete context_; context_ = nullptr; }
        if (engine_)  { delete engine_;  engine_  = nullptr; }
        if (runtime_) { delete runtime_; runtime_ = nullptr; }
    }

    bool IsReady() const { return context_ != nullptr && stream_ != nullptr; }

    // --------------------------------------------------------
    // Introspection
    // --------------------------------------------------------
    const std::vector<Binding>& Bindings() const { return bindings_; }
    int NumInputs()  const { return static_cast<int>(inputIdxs_.size()); }
    int NumOutputs() const { return static_cast<int>(outputIdxs_.size()); }

    const Binding& InputBinding(int i = 0)  const { return bindings_[inputIdxs_[i]]; }
    const Binding& OutputBinding(int i = 0) const { return bindings_[outputIdxs_[i]]; }

    // --------------------------------------------------------
    // Inference
    // --------------------------------------------------------

    // Convenience: single input, single output, float host vectors.
    // Handles FP16 conversion transparently — caller always sees float.
    // Resizes `output` to the actual output elemCount.
    bool Run(const std::vector<float>& input, std::vector<float>& output) {
        if (NumInputs() < 1 || NumOutputs() < 1) return false;
        const int inIdx  = inputIdxs_[0];
        const int outIdx = outputIdxs_[0];
        const auto& inB  = bindings_[inIdx];
        const auto& outB = bindings_[outIdx];

        if (input.size() != inB.elemCount) {
            std::cerr << "[TrtSession] Input size mismatch: got "
                      << input.size() << " expected " << inB.elemCount << "\n";
            return false;
        }
        output.resize(outB.elemCount);

        // Prepare input in native type.
        std::vector<__half> inHalf;
        const void* inPtr = input.data();
        if (inB.dataType == nvinfer1::DataType::kHALF) {
            inHalf.resize(inB.elemCount);
            for (size_t i = 0; i < inB.elemCount; ++i)
                inHalf[i] = __float2half(input[i]);
            inPtr = inHalf.data();
        }

        // Prepare output in native type.
        std::vector<__half> outHalf;
        void* outPtr = output.data();
        if (outB.dataType == nvinfer1::DataType::kHALF) {
            outHalf.resize(outB.elemCount);
            outPtr = outHalf.data();
        }

        if (!Dispatch({{inIdx, inPtr}}, {{outIdx, outPtr}})) return false;

        if (outB.dataType == nvinfer1::DataType::kHALF) {
            for (size_t i = 0; i < outB.elemCount; ++i)
                output[i] = __half2float(outHalf[i]);
        }
        return true;
    }

    // Generic: caller supplies (bindingIndex, host pointer) pairs.
    // Pointers must point to buffers of at least Binding.byteSize bytes,
    // in the binding's native data type (FP32 or FP16).
    bool Run(const std::vector<std::pair<int, const void*>>& inputs,
             const std::vector<std::pair<int, void*>>&       outputs) {
        return Dispatch(inputs, outputs);
    }

private:
    class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity sev, const char* msg) noexcept override {
            if (sev <= Severity::kWARNING) std::cerr << "[TRT] " << msg << "\n";
        }
    };

    bool BuildBindings() {
        const int nb = engine_->getNbBindings();
        if (nb <= 0) return false;

        bindings_.reserve(static_cast<size_t>(nb));
        for (int i = 0; i < nb; ++i) {
            Binding b;
            b.name     = engine_->getBindingName(i);
            b.isInput  = engine_->bindingIsInput(i);
            b.dims     = engine_->getBindingDimensions(i);
            b.dataType = engine_->getBindingDataType(i);

            switch (b.dataType) {
                case nvinfer1::DataType::kFLOAT: b.elemSize = sizeof(float);  break;
                case nvinfer1::DataType::kHALF:  b.elemSize = sizeof(__half); break;
                default:
                    std::cerr << "[TrtSession] Unsupported binding type for "
                              << b.name << " (only FP32/FP16 supported)\n";
                    return false;
            }

            size_t elems = 1;
            for (int d = 0; d < b.dims.nbDims; ++d) {
                if (b.dims.d[d] <= 0) {
                    std::cerr << "[TrtSession] Dynamic dim in binding "
                              << b.name << " (not supported)\n";
                    return false;
                }
                elems *= static_cast<size_t>(b.dims.d[d]);
            }
            b.elemCount = elems;
            b.byteSize  = elems * b.elemSize;

            bindings_.push_back(std::move(b));
            if (bindings_.back().isInput) inputIdxs_.push_back(i);
            else                          outputIdxs_.push_back(i);
        }

        if (inputIdxs_.empty() || outputIdxs_.empty()) {
            std::cerr << "[TrtSession] Engine must have at least one input and one output\n";
            return false;
        }
        return true;
    }

    bool AllocBuffers() {
        deviceBuffers_.assign(bindings_.size(), nullptr);
        for (size_t i = 0; i < bindings_.size(); ++i) {
            cudaError_t err = cudaMalloc(&deviceBuffers_[i], bindings_[i].byteSize);
            if (err != cudaSuccess) {
                std::cerr << "[TrtSession] cudaMalloc failed for " << bindings_[i].name
                          << ": " << cudaGetErrorString(err) << "\n";
                return false;
            }
        }
        return true;
    }

    bool Dispatch(const std::vector<std::pair<int, const void*>>& inputs,
                  const std::vector<std::pair<int, void*>>&       outputs) {
        if (!IsReady()) return false;
        try {
            for (const auto& in : inputs) {
                const int idx = in.first;
                cudaError_t err = cudaMemcpyAsync(
                    deviceBuffers_[idx], in.second, bindings_[idx].byteSize,
                    cudaMemcpyHostToDevice, stream_);
                if (err != cudaSuccess)
                    throw std::runtime_error(std::string("H2D[") +
                        bindings_[idx].name + "]: " + cudaGetErrorString(err));
            }

            if (!context_->enqueueV2(deviceBuffers_.data(), stream_, nullptr))
                throw std::runtime_error("enqueueV2 failed");

            for (const auto& out : outputs) {
                const int idx = out.first;
                cudaError_t err = cudaMemcpyAsync(
                    out.second, deviceBuffers_[idx], bindings_[idx].byteSize,
                    cudaMemcpyDeviceToHost, stream_);
                if (err != cudaSuccess)
                    throw std::runtime_error(std::string("D2H[") +
                        bindings_[idx].name + "]: " + cudaGetErrorString(err));
            }

            cudaError_t syncErr = cudaStreamSynchronize(stream_);
            if (syncErr != cudaSuccess)
                throw std::runtime_error(std::string("sync: ") + cudaGetErrorString(syncErr));
            return true;
        } catch (const std::exception& e) {
            std::cerr << "[TrtSession] Run error: " << e.what() << "\n";
            return false;
        }
    }

    // Declaration order matters: logger_ must outlive runtime_
    Logger               logger_;
    nvinfer1::IRuntime*  runtime_ = nullptr;
    nvinfer1::ICudaEngine*       engine_  = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;
    cudaStream_t         stream_  = nullptr;
    std::vector<void*>   deviceBuffers_;
    std::vector<Binding> bindings_;
    std::vector<int>     inputIdxs_;
    std::vector<int>     outputIdxs_;
};

}  // namespace perception
}  // namespace ucr