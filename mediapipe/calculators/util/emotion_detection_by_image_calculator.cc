// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "absl/log/absl_check.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "mediapipe/calculators/util/emotion_detection_by_image_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/detection.pb.h"
#include "mediapipe/framework/formats/location_data.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include <memory>
#include <vector>
#include <mutex>

// #define OUTPUT_RESULT_VALUE
// #define RENDER_CROP_FACE

namespace mediapipe {

namespace {

constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kDetectionsTag[] = "DETECTIONS";

// The ratio of detection label font height to the height of detection bounding
// box.
constexpr double kLabelToBoundingBoxRatio = 0.1;
// Perserve 2 decimal digits.
constexpr float kNumScoreDecimalDigitsMultipler = 100;

}  // namespace

// cv::VideoWriter writer;
class EmotionDetectionByImageCalculator : public CalculatorBase {
 public:
  EmotionDetectionByImageCalculator() {}
  ~EmotionDetectionByImageCalculator() override {}
  EmotionDetectionByImageCalculator(const EmotionDetectionByImageCalculator&) =
      delete;
  EmotionDetectionByImageCalculator& operator=(
      const EmotionDetectionByImageCalculator&) = delete;

  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;

private:
  struct TensorDetail {
      const char* name;
      TfLiteType type;
      TfLiteIntArray* dims;
  };
  absl::Status loadEmotionModel();
  void PrintTensorDetails() const;
  void PrintTensorDetail(const TensorDetail& detail) const;
  absl::Status CreateRenderTargetCpu(
      CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat,
      ImageFormat::Format* target_format);

  void dealWithDetection(const Detection& detection, cv::Mat& CVMat);
  cv::Rect calculateBBoxRect(const LocationData& location_data, int32_t width, int32_t height);
  void runTensorWithMat(const cv::Mat& face_mat);
private:

  EmotionDetectionByImageCalculatorOptions m_options;
  std::unique_ptr<::tflite::FlatBufferModel> m_model=nullptr;
  tflite::ops::builtin::BuiltinOpResolver m_resolver;
  std::unique_ptr<::tflite::Interpreter> m_interpreter; 
  std::vector<int> m_input_details;
  std::vector<int> m_output_details;
  std::vector<TensorDetail> m_input_tensor_details;
  std::vector<TensorDetail> m_output_tensor_details;
  int m_tf_num_thread=1;

  int32_t m_img_height=0;
  int32_t m_img_width=0;

  float m_emotion_threshold=0.9;
  int32_t m_last_id = 4;

#ifdef RENDER_CROP_FACE
  std::thread m_opencv_render_thread;
  std::mutex m_pMutex;
  cv::Mat m_will_render_mat;
#endif
};
REGISTER_CALCULATOR(EmotionDetectionByImageCalculator);

  absl::Status EmotionDetectionByImageCalculator::GetContract( CalculatorContract* cc) 
  {
    RET_CHECK(cc->Inputs().HasTag(kDetectionsTag))
        << "None of the input streams are provided.";

    if (cc->Inputs().HasTag(kImageFrameTag)) 
    {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
      printf("will process detections 0\n");
    }

    if (cc->Inputs().HasTag(kDetectionsTag)) 
    {
      cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
      printf("will process detections 1\n");
    }
    return absl::OkStatus();
  }

  absl::Status EmotionDetectionByImageCalculator::Open(CalculatorContext* cc) {
    cc->SetOffset(TimestampDiff(0));
    m_options = cc->Options<EmotionDetectionByImageCalculatorOptions>();
    ABSL_LOG(INFO) << "will load emotion model.";
    loadEmotionModel();
  #ifdef RENDER_CROP_FACE
    m_opencv_render_thread=std::thread([this](){
      cv::namedWindow("emotion detection", cv::WINDOW_AUTOSIZE);  
      cv::Mat frameBuff;
      while (true) {
        if (m_will_render_mat.empty())
        {
          cv::waitKey(50);
          continue;
        }
        if (m_pMutex.try_lock()) {
          frameBuff=m_will_render_mat;
          m_pMutex.unlock();
        }
        cv::imshow("emotion detection", m_will_render_mat);
        cv::waitKey(50);
      }
    });
  #endif

    return absl::OkStatus();
  }

  absl::Status EmotionDetectionByImageCalculator::Process(CalculatorContext* cc) 
  {
    if (cc->Inputs().Tag(kDetectionsTag).IsEmpty())
    {
      m_last_id = -1;//absent
      CalculatorGraph::CreateAndGetGlobaData()->emotion_type=(EEmotionType)m_last_id;
      return absl::OkStatus();
    }
    const bool has_detection_from_vector =
        cc->Inputs().HasTag(kDetectionsTag) &&
        !cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>().empty();
    
    if (!has_detection_from_vector) {
      return absl::OkStatus();
    }
    const auto& input_img = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
    if(m_img_width==0||m_img_height==0)
    {
      printf("input image width and height [%d %d] \n",
              input_img.Width(),input_img.Height());
      m_img_height = input_img.Height();
      m_img_width = input_img.Width();
    }
    cv::Mat CVMat = formats::MatView(&input_img);
    if (has_detection_from_vector)
    {
      for (const auto& detection : cc->Inputs().Tag(kDetectionsTag)
                                            .Get<std::vector<Detection>>()) 
      {
        dealWithDetection(detection, CVMat);
        break;
      }
    }
    return absl::OkStatus();
  }


  absl::Status EmotionDetectionByImageCalculator::loadEmotionModel(){
    if(!m_options.has_model_path())
    {
      ABSL_LOG(INFO) << "error: emotion mode path is empty.";
      return absl::InvalidArgumentError("model path is empty.");
    }
    ABSL_LOG(INFO) << "emotion mode path:"<<m_options.model_path().c_str();
    // Load the model
     m_model =
      ::tflite::FlatBufferModel::BuildFromFile(m_options.model_path().c_str());
    RET_CHECK(m_model) << "Failed to load TfLite model from model path.";
 
    // Build the m_interpreter
    tflite::InterpreterOptions options;
    options.SetPreserveAllTensors(false);
    options.SetDisableDelegateClustering(false);
    tflite::InterpreterBuilder builder(*m_model, m_resolver, &options);

    if(builder(&m_interpreter)==kTfLiteOk)
    {
      ABSL_LOG(INFO) << "m_interpreter build success.";
      std::cout<<"m_interpreter build success."<<std::endl;
      if(m_interpreter)
      {
        m_interpreter->SetNumThreads(m_tf_num_thread);
        // Resize input tensors, if desired.
        if(m_interpreter->AllocateTensors()!=kTfLiteOk)
        {
          ABSL_LOG(INFO) << "interpreter allocate tensor error.";
          printf("interpreter allocate tensor error.");
        }
        else
        {
          ABSL_LOG(INFO) << "model load finised.";
          printf("model load finised.\n");
          m_input_details = m_interpreter->inputs();
          
          for (int tensor_index : m_input_details) {
            
              const TfLiteTensor* tensor = m_interpreter->tensor(tensor_index);
              printf("input details name :%d %s %d %d %d\n",tensor_index,tensor->name, tensor->type, tensor->dims->size,tensor->bytes);
              m_input_tensor_details.push_back({tensor->name, tensor->type, tensor->dims});
          }

          m_output_details = m_interpreter->outputs();
          for (int tensor_index : m_output_details) {
              const TfLiteTensor* tensor = m_interpreter->tensor(tensor_index);
              printf("output details name :%d %s %d %d %d\n",tensor_index, tensor->name, tensor->type, tensor->dims->size,tensor->bytes);
              m_output_tensor_details.push_back({tensor->name, tensor->type, tensor->dims});
          }

          PrintTensorDetails();

          std::vector<int> output_indices_excluding_feedback_tensors;
          output_indices_excluding_feedback_tensors.reserve(
              m_interpreter->outputs().size());
          for (int i = 0; i < m_interpreter->outputs().size(); ++i) {
            output_indices_excluding_feedback_tensors.push_back(i);
          }
        }
      }
    }
    else
    {
      ABSL_LOG(INFO) << "m_interpreter build failure.";
      std::cout<<"m_interpreter build failure."<<std::endl;
    }
    return absl::OkStatus();
  }

  void EmotionDetectionByImageCalculator::PrintTensorDetails() const {
        std::cout << "Input Tensor Details:" << std::endl;
        for (const auto& detail : m_input_tensor_details) {
            PrintTensorDetail(detail);
        }

        std::cout << "Output Tensor Details:" << std::endl;
        for (const auto& detail : m_output_tensor_details) {
            PrintTensorDetail(detail);
        }
    }


  void EmotionDetectionByImageCalculator::PrintTensorDetail(const TensorDetail& detail) const {
      std::cout << "Name: " << detail.name << ", Type: " << detail.type << ", Dims: [";
      for (int i = 0; i < detail.dims->size; ++i) {
          std::cout << detail.dims->data[i];
          if (i < detail.dims->size - 1) {
              std::cout << ", ";
          }
      }
      std::cout << "]" << std::endl;
  }


absl::Status EmotionDetectionByImageCalculator::CreateRenderTargetCpu(
    CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat,
    ImageFormat::Format* target_format) {
  if (true) {
    const auto& input_frame =
        cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();

    int target_mat_type;
    switch (input_frame.Format()) {
      case ImageFormat::SRGBA:
        *target_format = ImageFormat::SRGBA;
        target_mat_type = CV_8UC4;
        break;
      case ImageFormat::SRGB:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      case ImageFormat::GRAY8:
        *target_format = ImageFormat::SRGB;
        target_mat_type = CV_8UC3;
        break;
      default:
        return absl::UnknownError("Unexpected image frame format.");
        break;
    }

    image_mat = absl::make_unique<cv::Mat>(
        input_frame.Height(), input_frame.Width(), target_mat_type);

    auto input_mat = formats::MatView(&input_frame);
    if (input_frame.Format() == ImageFormat::GRAY8) {
      cv::Mat rgb_mat;
      cv::cvtColor(input_mat, rgb_mat, cv::COLOR_GRAY2RGB);
      rgb_mat.copyTo(*image_mat);
    } else {
      input_mat.copyTo(*image_mat);
    }
  }
  return absl::OkStatus();
}


void EmotionDetectionByImageCalculator::dealWithDetection(const Detection& detection, 
                                                            cv::Mat& CVMat)
{
    auto matRect = calculateBBoxRect(detection.location_data(), m_img_width, m_img_height);
      
    // printf("leftTopX:%d leftTopY:%d width:%d height:%d\n",
              // leftTopX,leftTopY,rightBottomX-leftTopX,rightBottomY-leftTopY);
    //! 截取面部区域
    cv::Mat faceMat = CVMat(matRect);
    // 将面部区域调整为固定大小
    cv::Mat fixed_size_face;
    resize(faceMat, fixed_size_face, cv::Size(64, 64));
    // std::cout<<"mat channel 0: "<<faceMat.channels()<<std::endl;
    cv::cvtColor(fixed_size_face, fixed_size_face, cv::COLOR_BGR2RGB);
    // std::cout<<"mat channel 1: "<<fixed_size_face.channels()<<std::endl;
    // 进行检测
    fixed_size_face.convertTo(fixed_size_face, CV_32FC3, 1.0 / 255.0); // 归一化
#ifdef RENDER_CROP_FACE
    if (m_pMutex.try_lock()) {
      m_will_render_mat=fixed_size_face;
      m_pMutex.unlock();
    }
#endif
    runTensorWithMat(fixed_size_face);
}

  cv::Rect EmotionDetectionByImageCalculator::calculateBBoxRect(const LocationData& location_data, 
                                                            int32_t width, int32_t height)
  {
    auto &bboxC = location_data.relative_bounding_box();
    int32_t leftTopX = int(bboxC.xmin() * width);
    if(leftTopX<0) leftTopX=0;
    int32_t leftTopY = int(bboxC.ymin() * height);
    if(leftTopY<0) leftTopY=0;
    int32_t rightBottomX = int((bboxC.xmin() + bboxC.width()) * width);
    if(rightBottomX>width) rightBottomX=width;
    int32_t rightBottomY = int((bboxC.ymin() + bboxC.height()) * height);
    if (rightBottomY>height) rightBottomY=height;
    return cv::Rect(leftTopX, leftTopY, rightBottomX-leftTopX, rightBottomY-leftTopY);
  }

  void EmotionDetectionByImageCalculator::runTensorWithMat(const cv::Mat& face_mat)
  {
    // std::vector<float> input_data(face_mat.begin<float>(), face_mat.end<float>());
    // std::vector<cv::Mat> input_data = { face_mat };
    int input_details_tensor_index = m_input_details[0];
    int output_details_tensor_index = m_output_details[0];
    // Set the input tensor
        // 获取输入张量
    TfLiteTensor* input_tensor = m_interpreter->tensor(input_details_tensor_index);
    // float* input_tensor = m_interpreter->typed_tensor<float>(input_details_tensor_index);
    // std::copy(input_data.begin(), input_data.end(), input_tensor); //! 一维数据
    // 将输入数据赋值给模型输入//! 二维数据
    std::memcpy(input_tensor->data.f, face_mat.data, face_mat.total() * face_mat.elemSize());
    // 运行推理
    TfLiteStatus status_code = kTfLiteOk;
    tflite::Subgraph* subgraph = m_interpreter->subgraph(0);
    status_code = subgraph->Invoke();
    // std::cout<<"status code: "<<status_code<<std::endl;
    for (int tensor_index : subgraph->outputs()) {
      subgraph->EnsureTensorDataIsReadable(tensor_index);
    }
    // Get the output tensor
    auto tensor = m_interpreter->subgraph(0)->tensor(output_details_tensor_index);
    float* result = (float*)tensor->data.raw;
    // 处理输出数据
    // 找到最大值及其索引
    int num_output_elements = m_interpreter->tensor(output_details_tensor_index)->bytes / sizeof(float);
    // std::cout<<"num_output_elements: "<<num_output_elements<<std::endl;
    auto max_it = std::max_element(result, result + num_output_elements);
    float max_value = *max_it;
    int result_index = std::distance(result, max_it);
    // std::cout << "Predicted class: " << result_index << ", Probability: " << max_value << std::endl;
#ifdef OUTPUT_RESULT_VALUE
  for (size_t i = 0; i < num_output_elements; i++)
  {
    printf("result_index=%d value=%f \n",i,result[i]);
  }
#endif
    static int last_index=2;
    if (max_value >= m_emotion_threshold) {
        last_index = result_index;
        m_last_id = result_index;
    } else {
        m_last_id = result_index;
    }
    CalculatorGraph::CreateAndGetGlobaData()->emotion_type=(EEmotionType)m_last_id;

  }

}  // namespace mediapipe
