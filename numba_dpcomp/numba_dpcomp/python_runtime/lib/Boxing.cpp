// SPDX-FileCopyrightText: 2021 - 2022 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "PythonRt.hpp"

#include <memory>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#define SYCL_USM_ARRAY_INTERFACE "__sycl_usm_array_interface__"

namespace {
struct arystruct_t {
  void *meminfo; /* see _nrt_python.c and nrt.h in numba/core/runtime */
  PyObject *parent;
  npy_intp nitems;
  npy_intp itemsize;
  void *data;

  npy_intp shape_and_strides[1];
};

struct RefDeleter {
  template <typename T> void operator()(T *obj) const { Py_DECREF(obj); }
};
} // namespace

template <typename T> static std::unique_ptr<T, RefDeleter> makeRef(T *ref) {
  return std::unique_ptr<T, RefDeleter>(ref);
}

static bool initNumpy() {
  static bool init = []() -> bool {
    import_array1(false);
    return true;
  }();

  return init;
}

extern "C" DPCOMP_PYTHON_RUNTIME_EXPORT int
dpcompUnboxSyclInterface(PyObject *obj, arystruct_t *arystruct) {
  if (!initNumpy())
    return -1;

  auto iface = makeRef(PyObject_GetAttrString(obj, SYCL_USM_ARRAY_INTERFACE));
  if (!iface)
    return -1;

  auto data = [&]() -> void * {
    auto dataTuple = PyDict_GetItemString(iface.get(), "data");
    if (!dataTuple)
      return nullptr;

    auto item = PyTuple_GetItem(dataTuple, 0);
    if (!PyLong_Check(item))
      return nullptr;

    return PyLong_AsVoidPtr(item);
  }();

  if (!data)
    return -1;

  auto shapeObj = PyDict_GetItemString(iface.get(), "shape");
  if (!shapeObj)
    return -1;

  auto stridesObj = PyDict_GetItemString(iface.get(), "strides");
  if (!stridesObj)
    return -1;

  auto ndim = PyTuple_Size(shapeObj);
  if (ndim < 0)
    return -1;

  arystruct->data = data;
  arystruct->parent = obj;
  auto *dims = &arystruct->shape_and_strides[0];
  auto *strides = dims + ndim;

  npy_intp nitems = 1;
  for (decltype(ndim) i = 0; i < ndim; i++) {
    auto elem = PyTuple_GetItem(shapeObj, i);
    if (!elem || !PyLong_Check(elem))
      return -1;

    auto val = PyLong_AsLong(elem);
    nitems *= val;
    dims[i] = val;
  }
  auto itemsize = [&]() -> npy_intp {
    auto typestr = PyDict_GetItemString(iface.get(), "typestr");
    if (!typestr)
      return -1;

    PyArray_Descr *descr = nullptr;
    if (!PyArray_DescrConverter(typestr, &descr))
      return -1;

    auto descrRef = makeRef(descr);
    return descr->elsize;
  }();

  if (itemsize < 0)
    return -1;

  arystruct->itemsize = itemsize;
  arystruct->nitems = nitems;

  if (stridesObj == Py_None) {
    npy_intp stride = itemsize;
    for (decltype(ndim) i = 0; i < ndim; i++) {
      strides[ndim - i - 1] = stride;
      stride *= dims[ndim - i - 1];
    }
  } else {
    for (decltype(ndim) i = 0; i < ndim; i++) {
      auto elem = makeRef(PyTuple_GetItem(stridesObj, i));
      if (!elem || !PyLong_Check(elem.get()))
        return -1;

      strides[i] = PyLong_AsLong(elem.get());
    }
  }

  // TODO: dtor
  arystruct->meminfo =
      dpcompAllocMemInfo(data, itemsize * nitems, nullptr, nullptr);

  return 0;
}
