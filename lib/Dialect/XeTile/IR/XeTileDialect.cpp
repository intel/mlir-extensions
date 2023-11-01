#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdint>
#include <imex/Dialect/XeTile/IR/XeTileOps.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace xetile {

// bool TileBase::hasRank() const { return true; }

// llvm::ArrayRef<int64_t> TileBase::getShape() const {
//   return cast<TileType>().getShape();
// }

// TileBase TileBase::cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape,
//                              Type elementType, mlir::Attribute encoding)
//                              const {
//   return TileType::get(shape.value_or(getShape()), elementType, encoding);
// }

static mlir::LogicalResult
parseXeTileType(mlir::AsmParser &parser, llvm::SmallVector<int64_t> &shape,
                mlir::Type &type, llvm::SmallVector<int64_t> &inner_blocks,
                mlir::Attribute &xe_map) {
  mlir::Builder odsBuilder(parser.getContext());
  llvm::SMLoc odsLoc = parser.getCurrentLocation();
  (void)odsLoc;
  mlir::FailureOr<::imex::xetile::XeMapAttr> _result_xe_map;
  mlir::FailureOr<llvm::SmallVector<int64_t>> _result_inner_blocks;
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);

  bool shouldParseXeMap = true;

  if (mlir::succeeded(parser.parseOptionalComma())) {
    // try to parse 'inner_blocks'
    if (mlir::succeeded(parser.parseOptionalKeyword("inner_blocks"))) {
      if (parser.parseEqual())
        return mlir::failure();
      if (parser.parseLSquare())
        return mlir::failure();
      // Parse variable 'inner_blocks'
      _result_inner_blocks =
          mlir::FieldParser<llvm::SmallVector<int64_t>>::parse(parser);
      if (mlir::failed(_result_inner_blocks)) {
        parser.emitError(parser.getCurrentLocation(),
                         "failed to parse XeTile parameter 'inner_blocks' "
                         "which is to be a `llvm::ArrayRef<int64_t>`");
        return mlir::failure();
      }
      if (parser.parseRSquare())
        return mlir::failure();

      for (auto v : *_result_inner_blocks)
        inner_blocks.push_back(v);

      // check if there an additional comma, if so we need to parse XeMap
      if (mlir::failed(parser.parseOptionalComma())) {
        shouldParseXeMap = false;
      }
    }

    // Parse variable 'xe_map'
    if (shouldParseXeMap) {
      _result_xe_map =
          mlir::FieldParser<::imex::xetile::XeMapAttr>::parse(parser);
      if (mlir::failed(_result_xe_map)) {
        parser.emitError(
            parser.getCurrentLocation(),
            "failed to parse XeTile encoding parameter which is to "
            "be a `::imex::xetile::XeMapAttr`");
        return mlir::failure();
      }
      xe_map = std::move(_result_xe_map->dyn_cast<mlir::Attribute>());
    }
  }

  return mlir::success();
}

static void printXeTileType(mlir::AsmPrinter &printer,
                            llvm::ArrayRef<int64_t> shape, mlir::Type type,
                            llvm::ArrayRef<int64_t> inner_blocks,
                            mlir::Attribute xe_map) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;

  if (inner_blocks.size()) {
    printer << ", ";
    printer << "inner_blocks";
    printer << ' ' << "=";
    printer << ' ' << "[";
    printer.printStrippedAttrOrType(inner_blocks);
    printer << "]";
  }

  if (xe_map) {
    printer << ", ";
    printer.printStrippedAttrOrType(xe_map);
  }
}

mlir::LogicalResult
TileType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                 llvm::ArrayRef<int64_t> inner_blocks,
                 mlir::Attribute encoding) {
  if (inner_blocks.size() > 0 && inner_blocks.size() != 2)
    emitError() << "expect integer array of size 2 for inner_blocks";

  if (encoding) {
    auto xeMap = llvm::dyn_cast<imex::xetile::XeMapAttr>(encoding);
    if (!xeMap)
      emitError() << "expect xetile::XeMapAttr for encoding";
  }

  return mlir::success();
}

mlir::LogicalResult SubGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::DenseI32ArrayAttr mma_block_size, mlir::DenseI32ArrayAttr wi_layout,
    mlir::DenseI32ArrayAttr wi_data) {
  if (mma_block_size && mma_block_size.size() != 2)
    emitError() << "expect integer array of size 2 for mma_block_size";
  if (wi_layout.size() != 2)
    emitError() << "expect integer array of size 2 for wi_layout";
  if (wi_data.size() != 2)
    emitError() << "expect integer array of size 2 for wi_data";
  return mlir::success();
}

mlir::LogicalResult WorkGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::DenseI32ArrayAttr sg_layout, mlir::DenseI32ArrayAttr sg_data) {
  if (sg_layout.size() != 2)
    emitError() << "expect integer array of size 2 for sg_layout";
  if (sg_data.size() != 2)
    emitError() << "expect integer array of size 2 for sg_data";
  return mlir::success();
}

void XeTileDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/XeTile/IR/XeTileOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/XeTile/IR/XeTileOps.cpp.inc>
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include <imex/Dialect/XeTile/IR/XeTileOpsAttrs.cpp.inc>
      >();
}

} // namespace xetile
} // namespace imex

#include <imex/Dialect/XeTile/IR/XeTileOpsDialect.cpp.inc>
#define GET_ATTRDEF_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOpsAttrs.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/XeTile/IR/XeTileOpsTypes.cpp.inc>
