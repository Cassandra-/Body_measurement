# The set of languages for which implicit dependencies are needed:
SET(CMAKE_DEPENDS_LANGUAGES
  )
# The set of files for implicit dependencies of each language:

# Preprocessor definitions for this target.
SET(CMAKE_TARGET_DEFINITIONS
  "EIGEN_USE_NEW_STDVECTOR"
  "EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET"
  "QT_CORE_LIB"
  "QT_GUI_LIB"
  "QT_NO_DEBUG"
  "qh_QHpointer"
  )

# Targets to which this target links.
SET(CMAKE_TARGET_LINKED_INFO_FILES
  )

# The include file search paths:
SET(CMAKE_C_TARGET_INCLUDE_PATH
  "/usr/include/eigen3"
  "/usr/include/ni"
  "recognition/include/pcl/recognition/3rdparty"
  "/usr/include/qt4"
  "/usr/include/qt4/QtGui"
  "/usr/include/qt4/QtCore"
  "include"
  "common/include"
  "features/include"
  "filters/include"
  "geometry/include"
  "gpu/containers/include"
  "gpu/utils/include"
  "io/include"
  "kdtree/include"
  "octree/include"
  "search/include"
  "segmentation/include"
  "surface/include"
  "visualization/include"
  "sample_consensus/include"
  "people/include"
  "/usr/include/vtk-5.8"
  "gpu/people/include"
  "gpu/people/src"
  "gpu/people/src/cuda"
  "gpu/people/src/cuda/nvidia"
  )
SET(CMAKE_CXX_TARGET_INCLUDE_PATH ${CMAKE_C_TARGET_INCLUDE_PATH})
SET(CMAKE_Fortran_TARGET_INCLUDE_PATH ${CMAKE_C_TARGET_INCLUDE_PATH})
SET(CMAKE_ASM_TARGET_INCLUDE_PATH ${CMAKE_C_TARGET_INCLUDE_PATH})
