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

static mlir::LogicalResult parseXeTileType(mlir::AsmParser &parser,
                                           llvm::SmallVector<int64_t> &shape,
                                           mlir::Type &type,
                                           mlir::Attribute &tile_attr) {
  mlir::Builder odsBuilder(parser.getContext());
  llvm::SMLoc odsLoc = parser.getCurrentLocation();
  (void)odsLoc;
  mlir::FailureOr<::imex::xetile::XeTileAttr> _result_tile_attr;
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);

  if (mlir::succeeded(parser.parseOptionalComma())) {
    _result_tile_attr =
        mlir::FieldParser<::imex::xetile::XeTileAttr>::parse(parser);
    if (mlir::failed(_result_tile_attr)) {
      parser.emitError(parser.getCurrentLocation(),
                       "failed to parse XeTile encoding parameter which is to "
                       "be a `::imex::xetile::XeTileAttr`");
      return mlir::failure();
    }
    tile_attr = std::move(mlir::dyn_cast<mlir::Attribute>(*_result_tile_attr));
  }

  return mlir::success();
}

static void printXeTileType(mlir::AsmPrinter &printer,
                            llvm::ArrayRef<int64_t> shape, mlir::Type type,
                            mlir::Attribute tile_attr) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;

  if (tile_attr) {
    printer << ", ";
    printer.printStrippedAttrOrType(tile_attr);
  }
}

mlir::LogicalResult
TileType::verify(llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
                 llvm::ArrayRef<int64_t> shape, mlir::Type elementType,
                 mlir::Attribute encoding) {
  if (encoding) {
    auto xeTileAttr = llvm::dyn_cast<imex::xetile::XeTileAttr>(encoding);
    if (!xeTileAttr)
      emitError() << "expect xetile::XeTileAttr for encoding";
  }

  return mlir::success();
}

mlir::LogicalResult SubGroupMapAttr::verify(
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError,
    mlir::DenseI32ArrayAttr wi_layout, mlir::DenseI32ArrayAttr wi_data) {
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

mlir::LogicalResult XeTileAttr::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    ::imex::xetile::SubGroupMapAttr sg_map, xetile::WorkGroupMapAttr wg_map,
    mlir::DenseI32ArrayAttr order, mlir::DenseI64ArrayAttr inner_blocks,
    mlir::Attribute memoryScope) {

  if (order != mlir::DenseI32ArrayAttr() && order.size() != 2)
    emitError() << "expect integer array of size 2 for order";
  if (inner_blocks != mlir::DenseI64ArrayAttr() &&
      (inner_blocks.size() > 0 && inner_blocks.size() != 2))
    emitError() << "expect integer array of size 2 for non empty inner_blocks "
                   "attribute";
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
