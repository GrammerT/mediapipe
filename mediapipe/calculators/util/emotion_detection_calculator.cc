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

#include <memory>

#include "absl/log/absl_log.h"
#include "absl/strings/str_cat.h"
#include "mediapipe/calculators/util/emotion_detection_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "mediapipe/framework/formats/image.h"
#include "mediapipe/framework/formats/image_format.pb.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/formats/image_opencv.h"
#include "mediapipe/framework/formats/video_stream_header.h"
#include "mediapipe/framework/port/opencv_core_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/port/vector.h"
#include "mediapipe/util/annotation_renderer.h"
#include "mediapipe/util/color.pb.h"
#include "mediapipe/util/render_data.pb.h"


// #define LOG_FUNCTION_INFO() printf("Function: %s, File: %s, Line: %d\n", __FUNCTION__, __FILE__, __LINE__)
#define LOG_FUNCTION_INFO()

// #define RENDER_RECT_AND_POINTS

namespace mediapipe {

namespace {

constexpr char kVectorTag[] = "VECTOR";
constexpr char kGpuBufferTag[] = "IMAGE_GPU";
constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kImageTag[] = "UIMAGE";  // Universal Image

enum { ATTRIB_VERTEX, ATTRIB_TEXTURE_POSITION, NUM_ATTRIBUTES };

// Round up n to next multiple of m.
size_t RoundUp(size_t n, size_t m) { return ((n + m - 1) / m) * m; }  // NOLINT

// When using GPU, this color will become transparent when the calculator
// merges the annotation overlay with the image frame. As a result, drawing in
// this color is not supported and it should be set to something unlikely used.
constexpr uchar kAnnotationBackgroundColor = 2;  // Grayscale value.

// Future Image type.
inline bool HasImageTag(mediapipe::CalculatorContext* cc) {
  return cc->Inputs().HasTag(kImageTag);
}
}  // namespace

//
class EmotionDetectionCalculator : public CalculatorBase {
 public:
  EmotionDetectionCalculator() = default;
  ~EmotionDetectionCalculator() override = default;

  static absl::Status GetContract(CalculatorContract* cc);

  // From Calculator.
  absl::Status Open(CalculatorContext* cc) override;
  absl::Status Process(CalculatorContext* cc) override;
  absl::Status Close(CalculatorContext* cc) override;

 private:
  absl::Status CreateRenderTargetCpu(CalculatorContext* cc,
                                     std::unique_ptr<cv::Mat>& image_mat,
                                     ImageFormat::Format* target_format);
  absl::Status CreateRenderTargetCpuImage(CalculatorContext* cc,
                                          std::unique_ptr<cv::Mat>& image_mat,
                                          ImageFormat::Format* target_format);

  absl::Status RenderToCpu(CalculatorContext* cc,
                           const ImageFormat::Format& target_format,
                           uchar* data_image);

  absl::Status loadEmotionModel();
private:

  // Options for the calculator.
  EmotionDetectionCalculatorOptions options_;

  // Underlying helper renderer library.
  std::unique_ptr<AnnotationRenderer> renderer_;

  // Indicates if image frame is available as input.
  bool image_frame_available_ = false;

  bool use_gpu_ = false;
  bool gpu_initialized_ = false;

};
REGISTER_CALCULATOR(EmotionDetectionCalculator);

absl::Status EmotionDetectionCalculator::GetContract(CalculatorContract* cc) {
  RET_CHECK_GE(cc->Inputs().NumEntries(), 1);
  LOG_FUNCTION_INFO();
  bool use_gpu = false;

  RET_CHECK(cc->Inputs().HasTag(kImageFrameTag) +
                cc->Inputs().HasTag(kGpuBufferTag) +
                cc->Inputs().HasTag(kImageTag) <=
            1);
  RET_CHECK(cc->Outputs().HasTag(kImageFrameTag) +
                cc->Outputs().HasTag(kGpuBufferTag) +
                cc->Outputs().HasTag(kImageTag) ==
            1);

  // Input image to render onto copy of. Should be same type as output.

  if (cc->Inputs().HasTag(kImageFrameTag)) {
    cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
    RET_CHECK(cc->Outputs().HasTag(kImageFrameTag));
  }

  if (cc->Inputs().HasTag(kImageTag)) {
    cc->Inputs().Tag(kImageTag).Set<mediapipe::Image>();
    RET_CHECK(cc->Outputs().HasTag(kImageTag));

  }

  // Data streams to render.
  for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
       ++id) {
    auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
    std::string tag = tag_and_index.first;
    if (tag == kVectorTag) {
      cc->Inputs().Get(id).Set<std::vector<RenderData>>();
    } else if (tag.empty()) {
      // Empty tag defaults to accepting a single object of RenderData type.
      cc->Inputs().Get(id).Set<RenderData>();
    }
  }

  // Rendered image. Should be same type as input.

  if (cc->Outputs().HasTag(kImageFrameTag)) {
    cc->Outputs().Tag(kImageFrameTag).Set<ImageFrame>();
  }
  if (cc->Outputs().HasTag(kImageTag)) {
    cc->Outputs().Tag(kImageTag).Set<mediapipe::Image>();
  }

  LOG_FUNCTION_INFO();
  return absl::OkStatus();
}

absl::Status EmotionDetectionCalculator::Open(CalculatorContext* cc) {

  LOG_FUNCTION_INFO();
  cc->SetOffset(TimestampDiff(0));
  options_ = cc->Options<EmotionDetectionCalculatorOptions>();
  if (cc->Inputs().HasTag(kGpuBufferTag) ||
      cc->Inputs().HasTag(kImageFrameTag) || HasImageTag(cc)) {
    image_frame_available_ = true;
  } 

  loadEmotionModel();

#ifdef RENDER_RECT_AND_POINTS
  // Initialize the helper renderer library.
  renderer_ = absl::make_unique<AnnotationRenderer>();
  renderer_->SetFlipTextVertically(options_.flip_text_vertically());
  if (renderer_->GetScaleFactor() < 1.0 && HasImageTag(cc))
    ABSL_LOG(WARNING) << "Annotation scale factor only supports GPU backed Image.";
#endif
  // Set the output header based on the input header (if present).
  const char* tag = HasImageTag(cc) ? kImageTag
                    : use_gpu_      ? kGpuBufferTag
                                    : kImageFrameTag;
  if (image_frame_available_ && !cc->Inputs().Tag(tag).Header().IsEmpty()) {
    const auto& input_header =
        cc->Inputs().Tag(tag).Header().Get<VideoHeader>();
    auto* output_video_header = new VideoHeader(input_header);
    cc->Outputs().Tag(tag).SetHeader(Adopt(output_video_header));
  }
  LOG_FUNCTION_INFO();
  return absl::OkStatus();
}

absl::Status EmotionDetectionCalculator::Process(CalculatorContext* cc) 
{
  LOG_FUNCTION_INFO();
  if (cc->Inputs().HasTag(kImageFrameTag) &&
      cc->Inputs().Tag(kImageFrameTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (cc->Inputs().HasTag(kImageTag) && cc->Inputs().Tag(kImageTag).IsEmpty()) {
    return absl::OkStatus();
  }
  if (HasImageTag(cc)) {
    use_gpu_ = cc->Inputs().Tag(kImageTag).Get<mediapipe::Image>().UsesGpu();
  }

  // Initialize render target, drawn with OpenCV.
  std::unique_ptr<cv::Mat> image_mat;
  ImageFormat::Format target_format;
  //! only cpu
  if (cc->Outputs().HasTag(kImageTag)) {
    MP_RETURN_IF_ERROR(
        CreateRenderTargetCpuImage(cc, image_mat, &target_format));
  }
  if (cc->Outputs().HasTag(kImageFrameTag)) {
    MP_RETURN_IF_ERROR(CreateRenderTargetCpu(cc, image_mat, &target_format));
  }
#ifdef RENDER_RECT_AND_POINTS
  // Reset the renderer with the image_mat. No copy here.
  renderer_->AdoptImage(image_mat.get());
#endif

  // Render streams onto render target.
  for (CollectionItemId id = cc->Inputs().BeginId(); id < cc->Inputs().EndId();
       ++id) {
    auto tag_and_index = cc->Inputs().TagAndIndexFromId(id);
    std::string tag = tag_and_index.first;
    /**
     * node {
        calculator: "EmotionDetectionCalculator"
        input_stream: "IMAGE:input_image"
        input_stream: "detections_render_data"
        input_stream: "VECTOR:0:multi_face_landmarks_render_data"
        input_stream: "rects_render_data"
        output_stream: "IMAGE:output_image"
      }
     * 
     */
    // printf("annotation tag: %s \n", tag.c_str()); //! 会打印IMAGE 和 VECTOR
    if (!tag.empty() && tag != kVectorTag) {
      continue;//! IMAGE 会走此处
    }

    /**
     * 处理以下三个
      input_stream: "detections_render_data"
      input_stream: "VECTOR:0:multi_face_landmarks_render_data"
      input_stream: "rects_render_data"
     */

    if (cc->Inputs().Get(id).IsEmpty()) {
      continue; 
    }
    if (tag.empty()) {
      // Empty tag defaults to accepting a single object of RenderData type.
      const RenderData& render_data = cc->Inputs().Get(id).Get<RenderData>();
#ifdef RENDER_RECT_AND_POINTS
      renderer_->RenderDataOnImage(render_data);//! face mesh 中渲染 roi of face
#endif
    } else {
      RET_CHECK_EQ(kVectorTag, tag);
      const std::vector<RenderData>& render_data_vec =
          cc->Inputs().Get(id).Get<std::vector<RenderData>>();
      for (const RenderData& render_data : render_data_vec) {
#ifdef RENDER_RECT_AND_POINTS
        renderer_->RenderDataOnImage(render_data);//! face mesh中渲染所有landmark点 
#endif
      }
    }
  }
  // Copy the rendered image to output.
  uchar* image_mat_ptr = image_mat->data;
#ifdef RENDER_RECT_AND_POINTS
  MP_RETURN_IF_ERROR(RenderToCpu(cc, target_format, image_mat_ptr));
#else
  auto output_frame = absl::make_unique<ImageFrame>(target_format, image_mat->cols, image_mat->rows);
  output_frame->CopyPixelData(target_format, 
                                image_mat->cols,
                                image_mat->rows,
                                image_mat_ptr,
                                ImageFrame::kDefaultAlignmentBoundary);
  if (cc->Outputs().HasTag(kImageFrameTag)) {
    cc->Outputs()
        .Tag(kImageFrameTag)
        .Add(output_frame.release(), cc->InputTimestamp());
  }
#endif
  LOG_FUNCTION_INFO();
  return absl::OkStatus();
}

absl::Status EmotionDetectionCalculator::Close(CalculatorContext* cc) {
  LOG_FUNCTION_INFO();
  return absl::OkStatus();
}

absl::Status EmotionDetectionCalculator::RenderToCpu(
    CalculatorContext* cc, const ImageFormat::Format& target_format,
    uchar* data_image) {
    LOG_FUNCTION_INFO();

  auto output_frame = absl::make_unique<ImageFrame>(target_format, renderer_->GetImageWidth(), renderer_->GetImageHeight());

  output_frame->CopyPixelData(target_format, renderer_->GetImageWidth(),
                              renderer_->GetImageHeight(), data_image,
                              ImageFrame::kDefaultAlignmentBoundary);

  if (HasImageTag(cc)) {
    auto out = std::make_unique<mediapipe::Image>(std::move(output_frame));
    cc->Outputs().Tag(kImageTag).Add(out.release(), cc->InputTimestamp());
  }
  if (cc->Outputs().HasTag(kImageFrameTag)) {
    cc->Outputs()
        .Tag(kImageFrameTag)
        .Add(output_frame.release(), cc->InputTimestamp());
  }
  return absl::OkStatus();
}


absl::Status EmotionDetectionCalculator::loadEmotionModel()
{
  if(!options_.has_model_path())
  {
    printf("error: emotion mode path is empty.\n");
    return absl::InvalidArgumentError("model path is empty.");
  }

  printf("emotion mode path:%s \n",options_.model_path().c_str());
  return absl::OkStatus();
}

absl::Status EmotionDetectionCalculator::CreateRenderTargetCpu(
    CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat,
    ImageFormat::Format* target_format) {
      LOG_FUNCTION_INFO();
  if (image_frame_available_) {
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
  LOG_FUNCTION_INFO();
  return absl::OkStatus();
}

absl::Status EmotionDetectionCalculator::CreateRenderTargetCpuImage(
    CalculatorContext* cc, std::unique_ptr<cv::Mat>& image_mat,
    ImageFormat::Format* target_format) {
    LOG_FUNCTION_INFO();
  if (image_frame_available_) 
  {
    const auto& input_frame =
        cc->Inputs().Tag(kImageTag).Get<mediapipe::Image>();

    int target_mat_type;
    switch (input_frame.image_format()) {
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
        input_frame.height(), input_frame.width(), target_mat_type);

    auto input_mat = formats::MatView(&input_frame);
    if (input_frame.image_format() == ImageFormat::GRAY8) 
    {
      cv::Mat rgb_mat;
      cv::cvtColor(*input_mat, rgb_mat, cv::COLOR_GRAY2RGB);
      rgb_mat.copyTo(*image_mat);
    } 
    else 
    {
      input_mat->copyTo(*image_mat);
    }
  } 
  
  LOG_FUNCTION_INFO();
  return absl::OkStatus();
}


}  // namespace mediapipe
