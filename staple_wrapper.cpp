#include "staple_tracker.hpp"

#include <iostream>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;
using namespace boost::python;

namespace p = boost::python;
namespace np = boost::python::numpy;

class Staple {
    public:
    Staple() {
        Py_Initialize();
        np::initialize();

        tracker = new STAPLE_TRACKER();
    };

    ~Staple() {
        delete(tracker);
    };

    void init(np::ndarray image, np::ndarray bbox) {
        total_time = 0.0;
        frame_count = 0;

        int64 tic, toc;
        tic = cv::getTickCount();

        // convert image from ndarray to cv::Mat
        cv::Mat im = cv::Mat(
            image.shape(0),
            image.shape(1),
            CV_8UC3,
            image.get_data()
        );

		// convert bbox from ndarray to cv::Rect_<float>
		double* bboxData = (double*)bbox.get_data();
        cv::Rect_<float> region(
            float(bboxData[0]),
            float(bboxData[1]),
            float(bboxData[2] - bboxData[0]),
            float(bboxData[3] - bboxData[1])
        );

        /*
        cv::rectangle(im, region, cv::Scalar(0, 128, 255), 2);
        cv::imshow("update", im);
        cv::waitKey(0);
        */

        tracker->tracker_staple_initialize(im, region);
        tracker->tracker_staple_train(im, true);

        toc = cv::getTickCount() - tic;
        total_time += toc;
        frame_count++;
    };

    np::ndarray update(np::ndarray image) {
        int64 tic, toc;
        tic = cv::getTickCount();

        // convert image from ndarray to cv::Mat
        cv::Mat im = cv::Mat(
            image.shape(0),
            image.shape(1),
            CV_8UC3,
            image.get_data()
        );

        cv::Rect_<float> region = tracker->tracker_staple_update(im);
        update_response = tracker->get_last_response();

        /*
        cv::rectangle(im, region, cv::Scalar(0, 128, 255), 2);
        cv::imshow("update", im);
        cv::waitKey(0);
        */
       
        tracker->tracker_staple_train(im, false);

        // convert region to ndarray bbox - put bboxArray to heap
		// TODO : how to prevent memory leak?
		float* bboxArray;
		bboxArray = new float[4];
		bboxArray[0] = region.tl().x;
		bboxArray[1] = region.tl().y;
		bboxArray[2] = region.br().x;
		bboxArray[3] = region.br().y;
        
        np::ndarray bbox = np::from_data(
			bboxArray,
			np::dtype::get_builtin<float>(),
			p::make_tuple(4),
			p::make_tuple(sizeof(float)),
			p::object()
        );
        
        toc = cv::getTickCount() - tic;
        total_time += toc;
        frame_count++;

        return bbox;
    };

    double getTime() {
        return total_time / double(cv::getTickFrequency());
    }

    int getFrame() {
        return frame_count;
    }

    np::ndarray getResponse() {
		float* responseArray = (float*)update_response.data;
		np::ndarray response = np::from_data(
			responseArray,
			np::dtype::get_builtin<float>(),
			p::make_tuple(update_response.rows, update_response.cols),
			p::make_tuple(sizeof(float) * update_response.cols, sizeof(float)),
			p::object()
			);
        
        // response.reshape(p::make_tuple(update_response.rows, update_response.cols));

        return response;
    }

    private:
    STAPLE_TRACKER* tracker;

    double total_time = 0.0;
    int frame_count = 0;
    cv::Mat update_response;
};

BOOST_PYTHON_MODULE(StapleWrapper) {
    class_<Staple>("Staple", init<>())
        .def("init", &Staple::init)
        .def("update", &Staple::update)
        .add_property("time", &Staple::getTime)
        .add_property("frame", &Staple::getFrame)
        .add_property("response", &Staple::getResponse)
    ;
}
