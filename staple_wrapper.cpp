#include "staple_tracker.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

#include <opencv2/opencv.hpp>

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

		// draw im to check if it's ok
		cv::imshow("Staple", im);
		cv::waitKey(0);
		/*
		// convert bbox from ndarray to cv::Rect_<float>
		float* bboxData = (float*)bbox.get_data();
        cv::Rect_<float> region(
            float(bboxData[2] - bboxData[0] / 2.0),
            float(bboxData[3] - bboxData[1] / 2.0),
            float(bboxData[2] - bboxData[0]),
            float(bboxData[3] - bboxData[1])
        );

        tracker->tracker_staple_initialize(im, region);
        tracker->tracker_staple_train(im, true);
		*/
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

        // convert region to ndarray bbox
        float bboxArray[] = {
            region.tl().x,
            region.tl().y,
            region.br().x,
            region.br().y
        };
        
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
