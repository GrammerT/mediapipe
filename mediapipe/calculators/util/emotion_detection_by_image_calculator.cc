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
namespace mediapipe {

namespace {

constexpr char kDetectionTag[] = "DETECTION";
constexpr char kDetectionsTag[] = "DETECTIONS";
constexpr char kDetectionListTag[] = "DETECTION_LIST";
constexpr char kRenderDataTag[] = "RENDER_DATA";

constexpr char kSceneLabelLabel[] = "LABEL";
constexpr char kSceneFeatureLabel[] = "FEATURE";
constexpr char kSceneLocationLabel[] = "LOCATION";
constexpr char kKeypointLabel[] = "KEYPOINT";

// The ratio of detection label font height to the height of detection bounding
// box.
constexpr double kLabelToBoundingBoxRatio = 0.1;
// Perserve 2 decimal digits.
constexpr float kNumScoreDecimalDigitsMultipler = 100;

}  // namespace


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


};
REGISTER_CALCULATOR(EmotionDetectionByImageCalculator);

absl::Status EmotionDetectionByImageCalculator::GetContract(
    CalculatorContract* cc) {
  // RET_CHECK(cc->Inputs().HasTag(kDetectionListTag) ||
  //           cc->Inputs().HasTag(kDetectionsTag) ||
  //           cc->Inputs().HasTag(kDetectionTag))
  //     << "None of the input streams are provided.";

  // if (cc->Inputs().HasTag(kDetectionTag)) {
  //   cc->Inputs().Tag(kDetectionTag).Set<Detection>();
  // }
  // if (cc->Inputs().HasTag(kDetectionListTag)) {
  //   cc->Inputs().Tag(kDetectionListTag).Set<DetectionList>();
  // }
  // if (cc->Inputs().HasTag(kDetectionsTag)) {
  //   cc->Inputs().Tag(kDetectionsTag).Set<std::vector<Detection>>();
  // }
  // cc->Outputs().Tag(kRenderDataTag).Set<RenderData>();
  return absl::OkStatus();
}

absl::Status EmotionDetectionByImageCalculator::Open(CalculatorContext* cc) {
  // cc->SetOffset(TimestampDiff(0));

  return absl::OkStatus();
}

absl::Status EmotionDetectionByImageCalculator::Process(CalculatorContext* cc) 
{
  // const auto& options = cc->Options<DetectionsToRenderDataCalculatorOptions>();
  // const bool has_detection_from_list =
  //     cc->Inputs().HasTag(kDetectionListTag) && !cc->Inputs()
  //                                                    .Tag(kDetectionListTag)
  //                                                    .Get<DetectionList>()
  //                                                    .detection()
  //                                                    .empty();
  // const bool has_detection_from_vector =
  //     cc->Inputs().HasTag(kDetectionsTag) &&
  //     !cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>().empty();
  // const bool has_single_detection = cc->Inputs().HasTag(kDetectionTag) &&
  //                                   !cc->Inputs().Tag(kDetectionTag).IsEmpty();
  // if (!options.produce_empty_packet() && !has_detection_from_list &&
  //     !has_detection_from_vector && !has_single_detection) {
  //   return absl::OkStatus();
  // }

  // // TODO: Add score threshold to
  // // DetectionsToRenderDataCalculatorOptions.
  // auto render_data = absl::make_unique<RenderData>();
  // render_data->set_scene_class(options.scene_class());
  // if (has_detection_from_list) {
  //   for (const auto& detection :
  //        cc->Inputs().Tag(kDetectionListTag).Get<DetectionList>().detection()) {
  //     AddDetectionToRenderData(detection, options, render_data.get());
  //   }
  // }
  // if (has_detection_from_vector) {
  //   for (const auto& detection :
  //        cc->Inputs().Tag(kDetectionsTag).Get<std::vector<Detection>>()) {
  //     AddDetectionToRenderData(detection, options, render_data.get());
  //   }
  // }
  // if (has_single_detection) {
  //   AddDetectionToRenderData(cc->Inputs().Tag(kDetectionTag).Get<Detection>(),
  //                            options, render_data.get());
  // }
  // cc->Outputs()
  //     .Tag(kRenderDataTag)
  //     .Add(render_data.release(), cc->InputTimestamp());
  return absl::OkStatus();
}

}  // namespace mediapipe
