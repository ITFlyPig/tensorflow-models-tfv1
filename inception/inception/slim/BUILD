# Description:
#   Contains the operations and nets for building TensorFlow-Slim models.

package(default_visibility = ["//inception:internal"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "inception",
    srcs = ["inception_model.py"],
)

py_test(
    name = "inception_test",
    size = "medium",
    srcs = ["inception_test.py"],
    deps = [
        ":inception",
    ],
)

py_library(
    name = "slim",
    srcs = ["slim.py"],
    deps = [
        ":inception",
    ],
)

py_test(
    name = "collections_test",
    size = "small",
    srcs = ["collections_test.py"],
    deps = [
        ":slim",
    ],
)
