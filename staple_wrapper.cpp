#include "staple_tracker.hpp"

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

//namespace p = boost::python;
//namespace np = boost::python::numpy;

class Staple {
    public:
    Staple() {
        //Py_Initialize();
        //np::initialize();

        // tracker = new STAPLE_TRACKER();
    };

    ~Staple() {
        //delete(tracker);
    };

    void init(/*np::ndarray image, np::ndarray bbox*/) {
        // TODO : convert image from ndarray to cv::Mat

        // TODO : convert bbox from ndarray to cv::Rect_<float>

        //tracker.tracker_staple_initialize(im, region);
        //tracker.tracker_staple_train(im, true);
    };

    // np::ndarray update(np::ndarray image) {
        //cv::Rect region = tracker.tracker_staple_update(im);
        //tracker.tracker_staple_train(im, false);

        // TODO : convert bbox to ndarray

        //return bbox;
    //};

    private:
    //STAPLE_TRACKER* tracker;
};
/*
BOOST_PYTHON_MODULE(StapleWrapper) {
    class_<Staple>("Staple", init<>())
        .def("init", $Staple::init)
        .def("update", $Staple::update)
    ;
}
*/
