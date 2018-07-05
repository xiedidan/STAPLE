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

        tracker->tracker_staple_initialize(im, region);
        tracker->tracker_staple_train(im, true);
    };

    np::ndarray update(np::ndarray image) {
        // convert image from ndarray to cv::Mat
        cv::Mat im = cv::Mat(
            image.shape(0),
            image.shape(1),
            CV_8UC3,
            image.get_data()
        );

        cv::Rect_<float> region = tracker->tracker_staple_update(im);
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

        return bbox;
    };

    private:
    STAPLE_TRACKER* tracker;
};

BOOST_PYTHON_MODULE(StapleWrapper) {
    class_<Staple>("Staple", init<>())
        .def("init", &Staple::init)
        .def("update", &Staple::update)
    ;
}
