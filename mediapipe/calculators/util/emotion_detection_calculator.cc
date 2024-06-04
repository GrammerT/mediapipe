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

#include <cmath>
#include <functional>
#include <vector>
#include <algorithm>

#include "mediapipe/calculators/util/emotion_detection_calculator.pb.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/landmark.pb.h"
#include "mediapipe/framework/formats/rect.pb.h"
#include "mediapipe/framework/port/ret_check.h"
#include <tensorflow/lite/interpreter.h>
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/core/api/op_resolver.h"
#include "tensorflow/lite/kernels/register.h"
#include "mediapipe/framework/formats/image_frame.h"

namespace mediapipe {

using ::mediapipe::NormalizedRect;

namespace {

constexpr char kImageFrameTag[] = "IMAGE";
constexpr char kLandmarksTag[] = "NORM_LANDMARKS";
constexpr char kRectTag[] = "NORM_RECT";
constexpr char kProjectionMatrix[] = "PROJECTION_MATRIX";

}  // namespace

// Projects normalized landmarks to its original coordinates.
// Input:
//   NORM_LANDMARKS - NormalizedLandmarkList
//     Represents landmarks in a normalized rectangle if NORM_RECT is specified
//     or landmarks that should be projected using PROJECTION_MATRIX if
//     specified. (Prefer using PROJECTION_MATRIX as it eliminates need of
//     letterbox removal step.)
//   NORM_RECT - NormalizedRect
//     Represents a normalized rectangle in image coordinates and results in
//     landmarks with their locations adjusted to the image.
//   PROJECTION_MATRIX - std::array<float, 16>
//     A 4x4 row-major-order matrix that maps landmarks' locations from one
//     coordinate system to another. In this case from the coordinate system of
//     the normalized region of interest to the coordinate system of the image.
//
//   Note: either NORM_RECT or PROJECTION_MATRIX has to be specified.
//   Note: landmark's Z is projected in a custom way - it's scaled by width of
//     the normalized region of interest used during landmarks detection.
//
// Output:
//   NORM_LANDMARKS - NormalizedLandmarkList
//     Landmarks with their locations adjusted according to the inputs.
//
// Usage example:
// node {
//   calculator: "EmotionDetectionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "NORM_LANDMARKS:projected_landmarks"
// }
//
// node {
//   calculator: "EmotionDetectionCalculator"
//   input_stream: "NORM_LANDMARKS:0:landmarks_0"
//   input_stream: "NORM_LANDMARKS:1:landmarks_1"
//   input_stream: "NORM_RECT:rect"
//   output_stream: "NORM_LANDMARKS:0:projected_landmarks_0"
//   output_stream: "NORM_LANDMARKS:1:projected_landmarks_1"
// }
//
// node {
//   calculator: "EmotionDetectionCalculator"
//   input_stream: "NORM_LANDMARKS:landmarks"
//   input_stream: "PROECTION_MATRIX:matrix"
//   output_stream: "NORM_LANDMARKS:projected_landmarks"
// }
//
// node {
//   calculator: "EmotionDetectionCalculator"
//   input_stream: "NORM_LANDMARKS:0:landmarks_0"
//   input_stream: "NORM_LANDMARKS:1:landmarks_1"
//   input_stream: "PROECTION_MATRIX:matrix"
//   output_stream: "NORM_LANDMARKS:0:projected_landmarks_0"
//   output_stream: "NORM_LANDMARKS:1:projected_landmarks_1"
// }
class EmotionDetectionCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc) {

    RET_CHECK(cc->Inputs().HasTag(kImageFrameTag))
        << "Missing IMAGE input ,img width and height can't get.";

    RET_CHECK(cc->Inputs().HasTag(kLandmarksTag))
        << "Missing NORM_LANDMARKS input.";

    RET_CHECK_EQ(cc->Inputs().NumEntries(kLandmarksTag),
                 cc->Outputs().NumEntries(kLandmarksTag))
        << "Same number of input and output landmarks is required.";

    if (cc->Inputs().HasTag(kImageFrameTag)) {
      cc->Inputs().Tag(kImageFrameTag).Set<ImageFrame>();
    }

    for (CollectionItemId id = cc->Inputs().BeginId(kLandmarksTag);
         id != cc->Inputs().EndId(kLandmarksTag); ++id) {
      cc->Inputs().Get(id).Set<NormalizedLandmarkList>();
    }
    RET_CHECK(cc->Inputs().HasTag(kRectTag) ^
              cc->Inputs().HasTag(kProjectionMatrix))
        << "Either NORM_RECT or PROJECTION_MATRIX must be specified.";
    if (cc->Inputs().HasTag(kRectTag)) {
      cc->Inputs().Tag(kRectTag).Set<NormalizedRect>();
    } else {
      cc->Inputs().Tag(kProjectionMatrix).Set<std::array<float, 16>>();
    }

    for (CollectionItemId id = cc->Outputs().BeginId(kLandmarksTag);
         id != cc->Outputs().EndId(kLandmarksTag); ++id) {
      cc->Outputs().Get(id).Set<NormalizedLandmarkList>();
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    cc->SetOffset(TimestampDiff(0));
    m_options = cc->Options<EmotionDetectionCalculatorOptions>();
    loadEmotionModel();
    return absl::OkStatus();
  }

  static void ProjectXY(const NormalizedLandmark& lm,
                        const std::array<float, 16>& matrix,
                        NormalizedLandmark* out) {
    out->set_x(lm.x() * matrix[0] + lm.y() * matrix[1] + lm.z() * matrix[2] +
               matrix[3]);
    out->set_y(lm.x() * matrix[4] + lm.y() * matrix[5] + lm.z() * matrix[6] +
               matrix[7]);
  }

  /**
   * Landmark's Z scale is equal to a relative (to image) width of region of
   * interest used during detection. To calculate based on matrix:
   * 1. Project (0,0) --- (1,0) segment using matrix.
   * 2. Calculate length of the projected segment.
   */
  static float CalculateZScale(const std::array<float, 16>& matrix) {
    NormalizedLandmark a;
    a.set_x(0.0f);
    a.set_y(0.0f);
    NormalizedLandmark b;
    b.set_x(1.0f);
    b.set_y(0.0f);
    NormalizedLandmark a_projected;
    ProjectXY(a, matrix, &a_projected);
    NormalizedLandmark b_projected;
    ProjectXY(b, matrix, &b_projected);
    return std::sqrt(std::pow(b_projected.x() - a_projected.x(), 2) +
                     std::pow(b_projected.y() - a_projected.y(), 2));
  }

  absl::Status Process(CalculatorContext* cc) override {
    std::function<void(const NormalizedLandmark&, NormalizedLandmark*)>
        project_fn;
    if (cc->Inputs().HasTag(kRectTag)) {
      if (cc->Inputs().Tag(kRectTag).IsEmpty()) {
        return absl::OkStatus();
      }
      if(m_img_width==0||m_img_height==0)
      {
        const auto& input_img = cc->Inputs().Tag(kImageFrameTag).Get<ImageFrame>();
        printf("input image width and height [%d %d] \n",
                input_img.Width(),input_img.Height());
        m_img_height = input_img.Height();
        m_img_width = input_img.Width();
      }


      const auto& input_rect = cc->Inputs().Tag(kRectTag).Get<NormalizedRect>();
      const auto& options =
          cc->Options<mediapipe::EmotionDetectionCalculatorOptions>();
      project_fn = [&input_rect, &options](const NormalizedLandmark& landmark,
                                           NormalizedLandmark* new_landmark) {
        // TODO: fix projection or deprecate (current projection
        // calculations are incorrect for general case).
        const float x = landmark.x() - 0.5f;
        const float y = landmark.y() - 0.5f;
        const float angle =
            options.ignore_rotation() ? 0 : input_rect.rotation();
        float new_x = std::cos(angle) * x - std::sin(angle) * y;
        float new_y = std::sin(angle) * x + std::cos(angle) * y;

        new_x = new_x * input_rect.width() + input_rect.x_center();
        new_y = new_y * input_rect.height() + input_rect.y_center();
        const float new_z =
            landmark.z() * input_rect.width();  // Scale Z coordinate as X.

        *new_landmark = landmark;
        new_landmark->set_x(new_x);
        new_landmark->set_y(new_y);
        new_landmark->set_z(new_z);
      };
    } else if (cc->Inputs().HasTag(kProjectionMatrix)) {
      if (cc->Inputs().Tag(kProjectionMatrix).IsEmpty()) {
        return absl::OkStatus();
      }
      const auto& project_mat =
          cc->Inputs().Tag(kProjectionMatrix).Get<std::array<float, 16>>();
      const float z_scale = CalculateZScale(project_mat);
      project_fn = [&project_mat, z_scale](const NormalizedLandmark& lm,
                                           NormalizedLandmark* new_landmark) {
        *new_landmark = lm;
        ProjectXY(lm, project_mat, new_landmark);
        new_landmark->set_z(z_scale * lm.z());
      };
    } else {
      return absl::InternalError("Either rect or matrix must be specified.");
    }

    CollectionItemId input_id = cc->Inputs().BeginId(kLandmarksTag);
    CollectionItemId output_id = cc->Outputs().BeginId(kLandmarksTag);
    // Number of inputs and outpus is the same according to the contract.
    for (; input_id != cc->Inputs().EndId(kLandmarksTag);
         ++input_id, ++output_id) {
      const auto& input_packet = cc->Inputs().Get(input_id);
      if (input_packet.IsEmpty()) {
        continue;
      }

      const auto& input_landmarks = input_packet.Get<NormalizedLandmarkList>();
      NormalizedLandmarkList output_landmarks;
      for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
        const NormalizedLandmark& landmark = input_landmarks.landmark(i);
        NormalizedLandmark* new_landmark = output_landmarks.add_landmark();
        project_fn(landmark, new_landmark);
      }

      for (int i = 0; i < input_landmarks.landmark_size(); ++i) {
              const NormalizedLandmark& landmark = input_landmarks.landmark(i);
              NormalizedLandmark* new_landmark = output_landmarks.add_landmark();
              project_fn(landmark, new_landmark);
            }
#if 0
      printf("land mark content [ \n");
      for (int i = 0; i < output_landmarks.landmark_size(); ++i) {
              const NormalizedLandmark& landmark = output_landmarks.landmark(i);
              printf("x=%f,\n y=%f,\n z=%f \n",landmark.x(),landmark.y(),landmark.z());
            }
      printf("] \n");  
#endif
      cc->Outputs().Get(output_id).AddPacket(
          MakePacket<NormalizedLandmarkList>(std::move(output_landmarks))
              .At(cc->InputTimestamp()));
    }
    return absl::OkStatus();
  }

private:
  absl::Status loadEmotionModel()
  {
    if(!m_options.has_model_path())
    {
      printf("error: emotion mode path is empty.\n");
      return absl::InvalidArgumentError("model path is empty.");
    }

    printf("emotion mode path:%s \n",m_options.model_path().c_str());
// Load the model
    std::unique_ptr<::tflite::FlatBufferModel> model =
      ::tflite::FlatBufferModel::BuildFromFile(m_options.model_path().c_str());
    RET_CHECK(model) << "Failed to load TfLite model from model path.";
// Build the m_interpreter
    ::tflite::ops::builtin::BuiltinOpResolver resolver;
    ::tflite::InterpreterBuilder(*model, resolver)(&m_interpreter);

    RET_CHECK(m_interpreter);
    
    m_interpreter->SetNumThreads(m_tf_num_thread);
// Resize input tensors, if desired.
    m_interpreter->AllocateTensors();
    printf("model load finised.\n");
    m_interpreter->inputs();
    m_interpreter->outputs();
    return absl::OkStatus();
  }

private:
  // Options for the calculator.
  EmotionDetectionCalculatorOptions m_options;
  std::unique_ptr<::tflite::Interpreter> m_interpreter; 
  int m_tf_num_thread=1;

  int32_t m_img_width=0;
  int32_t m_img_height=0;
//   Angry
// Happy
// Neutral
// Sad
// Surprise
  std::vector<std::string> m_emotion_vec={"Angry","Happy","Neutral","Sad","Surprise"};
};
REGISTER_CALCULATOR(EmotionDetectionCalculator);

}  // namespace mediapipe
