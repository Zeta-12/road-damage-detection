// Road damage detection with an OpenVINO model exported by `rdd export`.
//
// rdd_detect --model best_int8_openvino_model/best.xml --input image_or_folder --output out_dir
//            [--device CPU|GPU|NPU] [--conf 0.25] [--iou 0.7]
//
// Writes the images with their boxes and detections.csv to out_dir, then prints the time of each step.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <openvino/openvino.hpp>

namespace fs = std::filesystem;

const std::vector<std::string> CLASSES = {"D00", "D10", "D20", "D40"};
const std::vector<cv::Scalar> COLORS = {{56, 56, 255}, {255, 112, 31}, {29, 178, 255}, {178, 29, 178}};  // BGR
const size_t MAX_DETECTIONS = 300;  // per image, like Ultralytics

struct Detection {
    int cls;
    float conf;
    cv::Rect2d box;  // pixels of the original image
};

// The image given to the model, and what is needed to put the boxes back on the original image
struct Letterbox {
    cv::Mat image;            // size x size
    double scale_x, scale_y;  // resized / original. Not quite equal: each side is rounded to whole pixels
    int left, top;            // grey border
};

// Same as Ultralytics: long side resized to `size`, then a grey border (114) up to size x size.
Letterbox letterbox(const cv::Mat& image, int size) {
    double r = std::min(double(size) / image.cols, double(size) / image.rows);
    int w = int(std::round(image.cols * r)), h = int(std::round(image.rows * r));
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);

    double dw = (size - w) / 2.0, dh = (size - h) / 2.0;
    int left = int(std::round(dw - 0.1)), right = int(std::round(dw + 0.1));
    int top = int(std::round(dh - 0.1)), bottom = int(std::round(dh + 0.1));
    Letterbox lb{cv::Mat(), double(w) / image.cols, double(h) / image.rows, left, top};
    cv::copyMakeBorder(resized, lb.image, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return lb;
}

// The model gives 1 x (4 + classes) x N: for each of the N anchors, the box (cx, cy, w, h) in pixels of the
// letterboxed image, then one score per class.
std::vector<Detection> postprocess(const ov::Tensor& output, const Letterbox& lb, cv::Size image_size, float conf,
                                   float iou) {
    const size_t channels = output.get_shape()[1], anchors = output.get_shape()[2];
    const float* data = output.data<const float>();
    auto at = [&](size_t channel, size_t anchor) { return data[channel * anchors + anchor]; };

    // 1. best class of each anchor, above the threshold
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> classes;
    for (size_t i = 0; i < anchors; i++) {
        int best = 0;
        for (int c = 1; c < int(channels) - 4; c++)
            if (at(4 + c, i) > at(4 + best, i)) best = c;
        float score = at(4 + best, i);
        if (score < conf) continue;
        boxes.emplace_back(at(0, i) - at(2, i) / 2, at(1, i) - at(3, i) / 2, at(2, i), at(3, i));
        scores.push_back(score);
        classes.push_back(best);
    }

    // 2. NMS per class
    std::vector<int> keep;  // sorted by score
    cv::dnn::NMSBoxesBatched(boxes, scores, classes, conf, iou, keep);
    if (keep.size() > MAX_DETECTIONS) keep.resize(MAX_DETECTIONS);

    // 3. back to the original image: remove the border, undo the resize, clip
    cv::Rect2d inside(0, 0, image_size.width, image_size.height);
    std::vector<Detection> detections;
    for (int k : keep) {
        const cv::Rect2d& b = boxes[k];
        cv::Rect2d box((b.x - lb.left) / lb.scale_x, (b.y - lb.top) / lb.scale_y, b.width / lb.scale_x,
                       b.height / lb.scale_y);
        detections.push_back({classes[k], scores[k], box & inside});
    }
    return detections;
}

void draw(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const Detection& d : detections) {
        cv::rectangle(image, cv::Rect(d.box), COLORS[d.cls], 2);
        std::string label = CLASSES[d.cls] + cv::format(" %.2f", d.conf);
        cv::Point origin(int(d.box.x), std::max(int(d.box.y) - 6, 14));
        cv::putText(image, label, origin, cv::FONT_HERSHEY_SIMPLEX, 0.6, COLORS[d.cls], 2);
    }
}

double milliseconds(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[v.size() / 2];
}

int main(int argc, char** argv) {
    std::map<std::string, std::string> args = {{"--device", "CPU"}, {"--conf", "0.25"}, {"--iou", "0.7"}};
    for (int i = 1; i + 1 < argc; i += 2) args[argv[i]] = argv[i + 1];
    if (!args.count("--model") || !args.count("--input") || !args.count("--output")) {
        std::cerr << "usage: rdd_detect --model model.xml --input image_or_folder --output out_dir"
                     " [--device CPU|GPU|NPU] [--conf 0.25] [--iou 0.7]\n";
        return 1;
    }
    float conf = std::stof(args["--conf"]), iou = std::stof(args["--iou"]);

    ov::Core core;
    ov::CompiledModel model = core.compile_model(args["--model"], args["--device"],
                                                 ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    ov::InferRequest request = model.create_infer_request();
    ov::Shape input_shape = model.input().get_shape();  // 1 x 3 x size x size
    int size = int(input_shape[2]);

    std::vector<fs::path> images;
    if (fs::is_directory(args["--input"])) {
        for (const auto& entry : fs::directory_iterator(args["--input"])) {
            std::string ext = entry.path().extension().string();
            if (ext == ".jpg" || ext == ".png") images.push_back(entry.path());
        }
        std::sort(images.begin(), images.end());
    } else {
        images.push_back(args["--input"]);
    }

    fs::path out = args["--output"];
    fs::create_directories(out);
    std::ofstream csv(out / "detections.csv");
    csv << "image,class,conf,x1,y1,x2,y2\n";

    std::vector<double> pre_ms, infer_ms, post_ms;
    for (const fs::path& path : images) {
        cv::Mat image = cv::imread(path.string());
        if (image.empty()) {
            std::cerr << "cannot read " << path << "\n";
            continue;
        }
        auto t0 = std::chrono::steady_clock::now();
        Letterbox lb = letterbox(image, size);
        // NCHW float, BGR to RGB, values in [0, 1]
        cv::Mat blob = cv::dnn::blobFromImage(lb.image, 1 / 255.0, cv::Size(), cv::Scalar(), true);

        auto t1 = std::chrono::steady_clock::now();
        request.set_input_tensor(ov::Tensor(ov::element::f32, input_shape, blob.ptr<float>()));
        request.infer();

        auto t2 = std::chrono::steady_clock::now();
        std::vector<Detection> detections = postprocess(request.get_output_tensor(), lb, image.size(), conf, iou);
        auto t3 = std::chrono::steady_clock::now();

        pre_ms.push_back(milliseconds(t0, t1));
        infer_ms.push_back(milliseconds(t1, t2));
        post_ms.push_back(milliseconds(t2, t3));
        for (const Detection& d : detections)
            csv << path.filename().string() << "," << CLASSES[d.cls] << "," << d.conf << "," << d.box.x << ","
                << d.box.y << "," << d.box.x + d.box.width << "," << d.box.y + d.box.height << "\n";
        draw(image, detections);
        cv::imwrite((out / path.filename()).string(), image);
    }

    std::cout << images.size() << " images on " << args["--device"] << ", median per image: preprocess "
              << median(pre_ms) << " ms, inference " << median(infer_ms) << " ms, postprocess " << median(post_ms)
              << " ms\n";
    return 0;
}
