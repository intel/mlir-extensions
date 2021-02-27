#ifndef PLIER_OPS
#define PLIER_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Plier_Dialect : Dialect {
  let name = "plier";
  let cppNamespace = "plier";
}

def Plier_PyType : DialectType<Plier_Dialect,
                 CPred<"$_self.isa<::plier::PyType>()">, "pytype">,
                 BuildableType<"$_builder.getType<::plier::PyType>()"> {
}

class Plier_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Plier_Dialect, mnemonic, traits>;

def ArgOp : Plier_Op<"arg", [NoSideEffect]> {
    let arguments = (ins
                     UI32Attr:$index,
                     StrAttr:$name);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilderDAG<(ins "unsigned":$index, "::mlir::StringRef":$name)>
    ];
}

def ConstOp : Plier_Op<"const", [NoSideEffect]> {
    let arguments = (ins
                     AnyAttr:$val);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Attribute":$val)>
    ];
}

def GlobalOp : Plier_Op<"global", [NoSideEffect]> {
    let arguments = (ins
                     StrAttr:$name);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::StringRef":$name)>
    ];
}

def BinOp : Plier_Op<"binop", []> {
    let arguments = (ins
                     AnyType:$lhs,
                     AnyType:$rhs,
                     StrAttr:$op);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$lhs, "::mlir::Value":$rhs, "::mlir::StringRef ":$op)>
    ];
}

def UnaryOp : Plier_Op<"unary", []> {
    let arguments = (ins
                     AnyType:$value,
                     StrAttr:$op);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value, "::mlir::StringRef ":$op)>
    ];
}

def CastOp : Plier_Op<"cast", []> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);
    let hasFolder = 1;
}

def PyCallOp : Plier_Op<"call", []> {
    let arguments = (ins
                     AnyType:$func,
                     Variadic<AnyType>:$args,
                     StrAttr:$func_name,
                     UI32Attr:$kw_start,
                     ArrayAttr:$kw_names);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$func, "::mlir::StringRef":$func_name,
                      "::mlir::ValueRange":$args,
                      "::mlir::ArrayRef<std::pair<std::string, ::mlir::Value>>":$kwargs)>
    ];
}

def BuildTupleOp : Plier_Op<"build_tuple", [NoSideEffect]> {
    let arguments = (ins
                     Variadic<AnyType>:$args);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::ValueRange":$args)>
    ];
}

def GetItemOp : Plier_Op<"getitem", []> {
    let arguments = (ins
                     AnyType:$value,
                     AnyType:$index);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value, "::mlir::Value":$index)>
    ];
}

def StaticGetItemOp : Plier_Op<"static_getitem", []> {
    let arguments = (ins
                     AnyType:$value,
                     AnyType:$index_var,
                     UI32Attr:$index);

    let results = (outs AnyType);
    let hasFolder = 1;

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value, "::mlir::Value":$index_var, "unsigned":$index)>
    ];
}

def SetItemOp : Plier_Op<"setitem", []> {
    let arguments = (ins
                     AnyType:$target,
                     AnyType:$index,
                     AnyType:$value);

    let builders = [];
}

def GetiterOp : Plier_Op<"getiter", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value)>
    ];
}

def IternextOp : Plier_Op<"iternext", []> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value)>
    ];
}

def PairfirstOp : Plier_Op<"pair_first", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value)>
    ];
}

def PairsecondOp : Plier_Op<"pair_second", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value);

    let results = (outs AnyType);

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value)>
    ];
}

def DelOp : Plier_Op<"del", []> {
    let arguments = (ins
                     AnyType:$value);
}

def GetattrOp : Plier_Op<"getattr", [NoSideEffect]> {
    let arguments = (ins
                     AnyType:$value,
                     StrAttr:$name);

    let results = (outs AnyType);

    let hasCanonicalizer = 1;

    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$value, "::mlir::StringRef":$name)>
    ];
}

def ParallelOp : Plier_Op<"parallel",
      [DeclareOpInterfaceMethods<LoopLikeOpInterface>,
       SingleBlockImplicitTerminator<"plier::YieldOp">,
       RecursiveSideEffects]> {

    let arguments = (ins Index:$lowerBound,
                     Index:$upperBound,
                     Index:$step);
    let regions = (region SizedRegion<1>:$region);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilderDAG<(ins "::mlir::Value":$lowerBound, "::mlir::Value":$upperBound, "::mlir::Value":$step,
            CArg<"::mlir::function_ref<void(::mlir::OpBuilder &, ::mlir::Location, ::mlir::Value, ::mlir::Value, ::mlir::Value)>",
                 "nullptr">)>
    ];
}

def YieldOp : Plier_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                                 ParentOneOf<["ParallelOp"]>]> {
    let arguments = (ins Variadic<AnyType>:$results);
    let builders = [OpBuilderDAG<(ins), [{ /* nothing to do */ }]>];
    // Override default verifier (defined in SCF_Op), no custom verification
    // needed.
    let verifier = ?;
}

#endif // PLIER_OPS
