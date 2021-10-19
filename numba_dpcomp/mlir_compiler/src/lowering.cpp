// Copyright 2021 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lowering.hpp"

#include <algorithm>
#include <array>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>
#include <mlir/Target/LLVMIR/Export.h>

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/Support/Debug.h>

#include "plier/dialect.hpp"

#include "plier/compiler/compiler.hpp"
#include "plier/compiler/pipeline_registry.hpp"
#include "plier/utils.hpp"

#include "pipelines/base_pipeline.hpp"
#include "pipelines/lower_to_gpu.hpp"
#include "pipelines/lower_to_llvm.hpp"
#include "pipelines/parallel_to_tbb.hpp"
#include "pipelines/plier_to_linalg.hpp"
#include "pipelines/plier_to_scf.hpp"
#include "pipelines/plier_to_std.hpp"
#include "pipelines/pre_low_simplifications.hpp"

namespace py = pybind11;
namespace {

class CallbackOstream : public llvm::raw_ostream {
public:
  using Func = std::function<void(llvm::StringRef)>;

  CallbackOstream(Func func = nullptr)
      : raw_ostream(/*unbuffered=*/false), callback(std::move(func)), pos(0u) {}

  ~CallbackOstream() override { flush(); }

  void write_impl(const char *ptr, size_t size) override {
    if (callback)
      callback(llvm::StringRef(ptr, size));
    pos += size;
  }

  uint64_t current_pos() const override { return pos; }

  void setCallback(Func func) { callback = std::move(func); }

private:
  Func callback;
  uint64_t pos;
};

std::string serialize_mod(const llvm::Module &mod) {
  std::string ret;
  llvm::raw_string_ostream stream(ret);
  llvm::WriteBitcodeToFile(mod, stream);
  stream.flush();
  return ret;
}

std::vector<std::pair<int, py::handle>> get_blocks(const py::object &func) {
  std::vector<std::pair<int, py::handle>> ret;
  auto blocks = func.attr("blocks").cast<py::dict>();
  ret.reserve(blocks.size());
  for (auto it : blocks) {
    ret.push_back({it.first.cast<int>(), it.second});
  }
  return ret;
}

py::list get_body(py::handle block) {
  return block.attr("body").cast<py::list>();
}

struct inst_handles {
  inst_handles() {
    auto mod = py::module::import("numba.core.ir");
    Assign = mod.attr("Assign");
    Del = mod.attr("Del");
    Return = mod.attr("Return");
    Branch = mod.attr("Branch");
    Jump = mod.attr("Jump");
    SetItem = mod.attr("SetItem");
    StaticSetItem = mod.attr("StaticSetItem");

    Arg = mod.attr("Arg");
    Expr = mod.attr("Expr");
    Var = mod.attr("Var");
    Const = mod.attr("Const");
    Global = mod.attr("Global");
    FreeVar = mod.attr("FreeVar");

    auto ops = py::module::import("operator");

    for (auto elem : llvm::zip(plier::getOperators(), ops_handles)) {
      auto name = std::get<0>(elem).name;
      std::get<1>(elem) = ops.attr(name.data());
    }
  }

  py::handle Assign;
  py::handle Del;
  py::handle Return;
  py::handle Branch;
  py::handle Jump;
  py::handle SetItem;
  py::handle StaticSetItem;

  py::handle Arg;
  py::handle Expr;
  py::handle Var;
  py::handle Const;
  py::handle Global;
  py::handle FreeVar;

  std::array<py::handle, plier::OperatorsCount> ops_handles;
};

struct plier_lowerer final {
  plier_lowerer(mlir::MLIRContext &context) : ctx(context), builder(&ctx) {
    ctx.loadDialect<mlir::StandardOpsDialect>();
    ctx.loadDialect<plier::PlierDialect>();
  }

  mlir::FuncOp lower(const py::object &compilation_context, mlir::ModuleOp mod,
                     const py::object &func_ir) {

    typemap = compilation_context["typemap"];
    func_name_resolver = compilation_context["resolve_func"];
    auto name = compilation_context["fnname"]().cast<std::string>();
    auto typ = get_func_type(compilation_context["fnargs"],
                             compilation_context["restype"]);
    func = mlir::FuncOp::create(builder.getUnknownLoc(), name, typ);
    if (compilation_context["fastmath"]().cast<bool>())
      func->setAttr(plier::attributes::getFastmathName(),
                    mlir::UnitAttr::get(&ctx));

    if (compilation_context["force_inline"]().cast<bool>())
      func->setAttr(plier::attributes::getForceInlineName(),
                    mlir::UnitAttr::get(&ctx));

    func->setAttr(plier::attributes::getOptLevelName(),
                  builder.getI64IntegerAttr(
                      compilation_context["opt_level"]().cast<int64_t>()));
    auto max_concurrency = compilation_context["max_concurrency"]().cast<int>();
    if (max_concurrency > 0) {
      mod->setAttr(plier::attributes::getMaxConcurrencyName(),
                   builder.getI64IntegerAttr(max_concurrency));
    }
    lower_func_body(func_ir);
    mod.push_back(func);
    return func;
  }

private:
  mlir::MLIRContext &ctx;
  mlir::OpBuilder builder;
  std::vector<mlir::Block *> blocks;
  std::unordered_map<int, mlir::Block *> blocks_map;
  inst_handles insts;
  mlir::FuncOp func;
  std::unordered_map<std::string, mlir::Value> vars_map;
  struct BlockInfo {
    struct PhiDesc {
      mlir::Block *dest_block = nullptr;
      std::string var_name;
      unsigned arg_index = 0;
    };
    llvm::SmallVector<PhiDesc, 2> outgoing_phi_nodes;
  };
  py::handle current_instr;
  py::handle typemap;
  py::handle func_name_resolver;

  std::unordered_map<mlir::Block *, BlockInfo> block_infos;

  plier::PyType get_obj_type(py::handle obj) const {
    return plier::PyType::get(&ctx, py::str(obj).cast<std::string>());
  }

  plier::PyType get_type(py::handle inst) const {
    auto type = typemap(inst);
    return get_obj_type(type);
  }

  void lower_func_body(const py::object &func_ir) {
    auto ir_blocks = get_blocks(func_ir);
    assert(!ir_blocks.empty());
    blocks.reserve(ir_blocks.size());
    for (std::size_t i = 0; i < ir_blocks.size(); ++i) {
      auto block = (0 == i ? func.addEntryBlock() : func.addBlock());
      blocks.push_back(block);
      blocks_map[ir_blocks[i].first] = block;
    }

    for (std::size_t i = 0; i < ir_blocks.size(); ++i) {
      lower_block(blocks[i], ir_blocks[i].second);
    }
    fixup_phis();
  }

  void lower_block(mlir::Block *bb, py::handle ir_block) {
    assert(nullptr != bb);
    builder.setInsertionPointToEnd(bb);
    for (auto it : get_body(ir_block)) {
      current_instr = it;
      lower_inst(it);
      current_instr = nullptr;
    }
  }

  void lower_inst(py::handle inst) {
    if (py::isinstance(inst, insts.Assign)) {
      auto target = inst.attr("target");
      auto val = lower_assign(inst, target);
      storevar(val, target);
    } else if (py::isinstance(inst, insts.SetItem) ||
               py::isinstance(inst, insts.StaticSetItem)) {
      setitem(inst.attr("target"), inst.attr("index"), inst.attr("value"));
    } else if (py::isinstance(inst, insts.Del)) {
      delvar(inst.attr("value"));
    } else if (py::isinstance(inst, insts.Return)) {
      retvar(inst.attr("value"));
    } else if (py::isinstance(inst, insts.Branch)) {
      branch(inst.attr("cond"), inst.attr("truebr"), inst.attr("falsebr"));
    } else if (py::isinstance(inst, insts.Jump)) {
      jump(inst.attr("target"));
    } else {
      plier::report_error(llvm::Twine("lower_inst not handled: \"") +
                          py::str(inst.get_type()).cast<std::string>() + "\"");
    }
  }

  mlir::Value lower_assign(py::handle inst, py::handle target) {
    auto value = inst.attr("value");
    if (py::isinstance(value, insts.Arg)) {
      auto index = value.attr("index").cast<std::size_t>();
      return builder.create<plier::ArgOp>(
          get_current_loc(), index, target.attr("name").cast<std::string>());
    }
    if (py::isinstance(value, insts.Expr)) {
      return lower_expr(value);
    }
    if (py::isinstance(value, insts.Var)) {
      return loadvar(value);
    }
    if (py::isinstance(value, insts.Const)) {
      return get_const(value.attr("value"));
    }
    if (py::isinstance(value, insts.Global) ||
        py::isinstance(value, insts.FreeVar)) {
      auto name = value.attr("name").cast<std::string>();
      return builder.create<plier::GlobalOp>(get_current_loc(), name);
    }

    plier::report_error(llvm::Twine("lower_assign not handled: \"") +
                        py::str(value.get_type()).cast<std::string>() + "\"");
  }

  mlir::Value lower_expr(py::handle expr) {
    auto op = expr.attr("op").cast<std::string>();
    using func_t = mlir::Value (plier_lowerer::*)(py::handle);
    const std::pair<mlir::StringRef, func_t> handlers[] = {
        {"binop", &plier_lowerer::lower_binop},
        {"inplace_binop", &plier_lowerer::lower_inplce_binop},
        {"unary", &plier_lowerer::lower_unary},
        {"cast", &plier_lowerer::lower_cast},
        {"call", &plier_lowerer::lower_call},
        {"phi", &plier_lowerer::lower_phi},
        {"build_tuple", &plier_lowerer::lower_build_tuple},
        {"getitem", &plier_lowerer::lower_getitem},
        {"static_getitem", &plier_lowerer::lower_static_getitem},
        {"getiter", &plier_lowerer::lower_simple<plier::GetiterOp>},
        {"iternext", &plier_lowerer::lower_simple<plier::IternextOp>},
        {"pair_first", &plier_lowerer::lower_simple<plier::PairfirstOp>},
        {"pair_second", &plier_lowerer::lower_simple<plier::PairsecondOp>},
        {"getattr", &plier_lowerer::lower_getattr},
        {"exhaust_iter", &plier_lowerer::lower_exhaust_iter},
    };
    for (auto &h : handlers) {
      if (h.first == op) {
        return (this->*h.second)(expr);
      }
    }
    plier::report_error(llvm::Twine("lower_expr not handled: \"") + op + "\"");
  }

  template <typename T> mlir::Value lower_simple(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    return builder.create<T>(get_current_loc(), value);
  }

  mlir::Value lower_cast(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto res_type = get_type(current_instr.attr("target"));
    return builder.create<plier::CastOp>(get_current_loc(), res_type, value);
  }

  mlir::Value lower_getitem(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto index = loadvar(inst.attr("index"));
    return builder.create<plier::GetItemOp>(get_current_loc(), value, index);
  }

  mlir::Value lower_static_index(mlir::Location loc, py::handle obj) {
    if (obj.is_none()) {
      auto type = mlir::NoneType::get(builder.getContext());
      return builder.create<plier::UndefOp>(loc, type);
    }
    if (py::isinstance<py::int_>(obj)) {
      auto index = obj.cast<int64_t>();
      return builder.create<mlir::arith::ConstantIndexOp>(loc, index);
    }
    if (py::isinstance<py::slice>(obj)) {
      auto start = lower_static_index(loc, obj.attr("start"));
      auto stop = lower_static_index(loc, obj.attr("stop"));
      auto step = lower_static_index(loc, obj.attr("step"));
      return builder.create<plier::BuildSliceOp>(loc, start, stop, step);
    }
    if (py::isinstance<py::iterable>(obj)) {
      llvm::SmallVector<mlir::Value> args(py::len(obj));
      for (auto it : llvm::enumerate(obj))
        args[it.index()] = lower_static_index(loc, it.value());

      return builder.create<plier::BuildTupleOp>(loc, args);
    }
    plier::report_error(llvm::Twine("Unhandled index type: ") +
                        py::str(obj.get_type()).cast<std::string>());
  }

  mlir::Value lower_static_getitem(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto loc = get_current_loc();
    auto index_var = lower_static_index(loc, inst.attr("index"));
    return builder.create<plier::GetItemOp>(loc, value, index_var);
  }

  mlir::Value lower_build_tuple(py::handle inst) {
    auto items = inst.attr("items").cast<py::list>();
    mlir::SmallVector<mlir::Value> args;
    for (auto item : items) {
      args.push_back(loadvar(item));
    }
    return builder.create<plier::BuildTupleOp>(get_current_loc(), args);
  }

  mlir::Value lower_phi(py::handle expr) {
    auto incoming_vals = expr.attr("incoming_values").cast<py::list>();
    auto incoming_blocks = expr.attr("incoming_blocks").cast<py::list>();
    assert(incoming_vals.size() == incoming_blocks.size());

    auto current_block = builder.getBlock();
    assert(nullptr != current_block);

    auto arg_index = current_block->getNumArguments();
    auto arg =
        current_block->addArgument(get_type(current_instr.attr("target")));

    auto count = incoming_vals.size();
    for (std::size_t i = 0; i < count; ++i) {
      auto var = incoming_vals[i].attr("name").cast<std::string>();
      auto block = blocks_map.find(incoming_blocks[i].cast<int>())->second;
      block_infos[block].outgoing_phi_nodes.push_back(
          {current_block, std::move(var), arg_index});
    }

    return arg;
  }

  mlir::Value lower_call(py::handle expr) {
    auto pyPunc = expr.attr("func");
    auto func = loadvar(pyPunc);
    auto args = expr.attr("args").cast<py::list>();
    auto kws = expr.attr("kws").cast<py::list>();
    auto vararg = expr.attr("vararg");

    auto varargVar = (vararg.is_none() ? mlir::Value() : loadvar(vararg));

    mlir::SmallVector<mlir::Value> argsList;
    argsList.reserve(args.size());
    for (auto a : args)
      argsList.push_back(loadvar(a));

    mlir::SmallVector<std::pair<std::string, mlir::Value>> kwargsList;
    for (auto a : kws) {
      auto item = a.cast<py::tuple>();
      auto name = item[0];
      auto valName = item[1];
      kwargsList.push_back({name.cast<std::string>(), loadvar(valName)});
    }

    auto pyFuncName = func_name_resolver(typemap(pyPunc));
    if (pyFuncName.is_none())
      plier::report_error(llvm::Twine("Can't resolve function: ") +
                          py::str(typemap(pyPunc)).cast<std::string>());

    auto funcName = pyFuncName.cast<std::string>();

    return builder.create<plier::PyCallOp>(get_current_loc(), func, funcName,
                                           argsList, varargVar, kwargsList);
  }

  mlir::Value lower_binop(py::handle expr) {
    auto op = expr.attr("fn");
    auto lhs_name = expr.attr("lhs");
    auto rhs_name = expr.attr("rhs");
    auto lhs = loadvar(lhs_name);
    auto rhs = loadvar(rhs_name);
    auto op_name = resolve_op(op);
    return builder.create<plier::BinOp>(get_current_loc(), lhs, rhs, op_name);
  }

  mlir::Value lower_inplce_binop(py::handle expr) {
    auto op = expr.attr("immutable_fn");
    auto lhs_name = expr.attr("lhs");
    auto rhs_name = expr.attr("rhs");
    auto lhs = loadvar(lhs_name);
    auto rhs = loadvar(rhs_name);
    auto op_name = resolve_op(op);
    return builder.create<plier::BinOp>(get_current_loc(), lhs, rhs, op_name);
  }

  mlir::Value lower_unary(py::handle expr) {
    auto op = expr.attr("fn");
    auto val_name = expr.attr("value");
    auto val = loadvar(val_name);
    auto op_name = resolve_op(op);
    return builder.create<plier::UnaryOp>(get_current_loc(), val, op_name);
  }

  llvm::StringRef resolve_op(py::handle op) {
    for (auto elem : llvm::zip(plier::getOperators(), insts.ops_handles)) {
      if (op.is(std::get<1>(elem))) {
        return std::get<0>(elem).op;
      }
    }

    plier::report_error(llvm::Twine("resolve_op not handled: \"") +
                        py::str(op).cast<std::string>() + "\"");
  }

  mlir::Value lower_getattr(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto name = inst.attr("attr").cast<std::string>();
    return builder.create<plier::GetattrOp>(get_current_loc(), value, name);
  }

  mlir::Value lower_exhaust_iter(py::handle inst) {
    auto value = loadvar(inst.attr("value"));
    auto count = inst.attr("count").cast<int64_t>();
    return builder.create<plier::ExhaustIterOp>(get_current_loc(), value,
                                                count);
  }

  void setitem(py::handle target, py::handle index, py::handle value) {
    auto ind = [&]() -> mlir::Value {
      if (py::isinstance<py::int_>(index))
        return builder.create<mlir::arith::ConstantIndexOp>(
            get_current_loc(), index.cast<int64_t>());

      return loadvar(index);
    }();
    builder.create<plier::SetItemOp>(get_current_loc(), loadvar(target), ind,
                                     loadvar(value));
  }

  void storevar(mlir::Value val, py::handle inst) {
    vars_map[inst.attr("name").cast<std::string>()] = val;
    val.setType(get_type(inst));
  }

  mlir::Value loadvar(py::handle inst) {
    auto it = vars_map.find(inst.attr("name").cast<std::string>());
    assert(vars_map.end() != it);
    return it->second;
  }

  void delvar(py::handle inst) {
    auto var = loadvar(inst);
    builder.create<plier::DelOp>(get_current_loc(), var);
  }

  void retvar(py::handle inst) {
    auto var = loadvar(inst);
    auto func_type = func.getType();
    auto ret_type = func_type.getResult(0);
    auto var_type = var.getType();
    if (ret_type != var_type)
      var = builder.create<plier::CastOp>(get_current_loc(), ret_type, var);

    builder.create<mlir::ReturnOp>(get_current_loc(), var);
  }

  void branch(py::handle cond, py::handle tr, py::handle fl) {
    auto c = loadvar(cond);
    auto tr_block = blocks_map.find(tr.cast<int>())->second;
    auto fl_block = blocks_map.find(fl.cast<int>())->second;
    auto cond_val = builder.create<plier::CastOp>(
        get_current_loc(), mlir::IntegerType::get(&ctx, 1), c);
    builder.create<mlir::CondBranchOp>(get_current_loc(), cond_val, tr_block,
                                       fl_block);
  }

  void jump(py::handle target) {
    auto block = blocks_map.find(target.cast<int>())->second;
    builder.create<mlir::BranchOp>(get_current_loc(), mlir::None, block);
  }

  mlir::Value get_const(py::handle val) {
    auto get_val = [&](mlir::Attribute attr) {
      return builder.create<plier::ConstOp>(get_current_loc(), attr);
    };
    if (py::isinstance<py::int_>(val)) {
      auto type = mlir::IntegerType::get(builder.getContext(), 64,
                                         mlir::IntegerType::Signed);
      auto attr = builder.getIntegerAttr(type, val.cast<int64_t>());
      return get_val(attr);
    }
    if (py::isinstance<py::float_>(val))
      return get_val(builder.getF64FloatAttr(val.cast<double>()));

    if (py::isinstance<py::none>(val))
      return get_val(builder.getUnitAttr());

    plier::report_error(llvm::Twine("get_const unhandled type \"") +
                        py::str(val.get_type()).cast<std::string>() + "\"");
  }

  mlir::FunctionType get_func_type(py::handle fnargs, py::handle restype) {
    auto ret = get_obj_type(restype());
    llvm::SmallVector<mlir::Type> args;
    for (auto arg : fnargs()) {
      args.push_back(get_obj_type(arg));
    }
    return mlir::FunctionType::get(&ctx, args, {ret});
  }

  mlir::Location get_current_loc() {
    return builder.getUnknownLoc(); // TODO
  }

  void fixup_phis() {
    auto build_arg_list = [&](mlir::Block *block, auto &outgoing_phi_nodes,
                              auto &list) {
      for (auto &o : outgoing_phi_nodes) {
        if (o.dest_block == block) {
          auto arg_index = o.arg_index;
          if (list.size() <= arg_index) {
            list.resize(arg_index + 1);
          }
          auto it = vars_map.find(o.var_name);
          assert(vars_map.end() != it);
          auto arg_type = block->getArgument(arg_index).getType();
          auto val = builder.create<plier::CastOp>(builder.getUnknownLoc(),
                                                   arg_type, it->second);
          list[arg_index] = val;
        }
      }
    };
    for (auto &bb : func) {
      auto it = block_infos.find(&bb);
      if (block_infos.end() != it) {
        auto &info = it->second;
        auto term = bb.getTerminator();
        if (nullptr == term) {
          plier::report_error("broken ir: block without terminator");
        }
        builder.setInsertionPointToEnd(&bb);

        if (auto op = mlir::dyn_cast<mlir::BranchOp>(term)) {
          auto dest = op.getDest();
          mlir::SmallVector<mlir::Value> args;
          build_arg_list(dest, info.outgoing_phi_nodes, args);
          op.erase();
          builder.create<mlir::BranchOp>(builder.getUnknownLoc(), dest, args);
        } else if (auto op = mlir::dyn_cast<mlir::CondBranchOp>(term)) {
          auto true_dest = op.trueDest();
          auto false_dest = op.falseDest();
          auto cond = op.getCondition();
          mlir::SmallVector<mlir::Value> true_args;
          mlir::SmallVector<mlir::Value> false_args;
          build_arg_list(true_dest, info.outgoing_phi_nodes, true_args);
          build_arg_list(false_dest, info.outgoing_phi_nodes, false_args);
          op.erase();
          builder.create<mlir::CondBranchOp>(builder.getUnknownLoc(), cond,
                                             true_dest, true_args, false_dest,
                                             false_args);
        } else {
          plier::report_error(llvm::Twine("Unhandled terminator: ") +
                              term->getName().getStringRef());
        }
      }
    }
  }
};

plier::CompilerContext::Settings getSettings(py::handle settings,
                                             CallbackOstream &os) {
  plier::CompilerContext::Settings ret;
  ret.verify = settings["verify"].cast<bool>();
  ret.passStatistics = settings["pass_statistics"].cast<bool>();
  ret.passTimings = settings["pass_timings"].cast<bool>();
  ret.irDumpStderr = settings["ir_printing"].cast<bool>();
  ret.diagDumpStderr = settings["diag_printing"].cast<bool>();

  auto printBefore = settings["print_before"].cast<py::list>();
  auto printAfter = settings["print_after"].cast<py::list>();
  if (!printBefore.empty() || !printAfter.empty()) {
    auto callback = settings["print_callback"].cast<py::function>();
    auto getList = [](py::list src) {
      llvm::SmallVector<std::string, 1> res(src.size());
      for (auto it : llvm::enumerate(src)) {
        res[it.index()] = py::str(it.value()).cast<std::string>();
      }
      return res;
    };
    os.setCallback([callback](llvm::StringRef text) {
      callback(py::str(text.data(), text.size()));
    });
    using S = plier::CompilerContext::Settings::IRPrintingSettings;
    ret.irPrinting = S{getList(printBefore), getList(printAfter), &os};
  }
  return ret;
}

py::bytes gen_ll_module(mlir::ModuleOp mod) {
  std::string err;
  llvm::raw_string_ostream errStream(err);
  auto diag_handler = [&](mlir::Diagnostic &diag) {
    if (diag.getSeverity() == mlir::DiagnosticSeverity::Error) {
      errStream << diag;
    }
  };
  llvm::LLVMContext llCtx;
  std::unique_ptr<llvm::Module> llMod;
  plier::scoped_diag_handler(*mod.getContext(), diag_handler, [&]() {
    mlir::registerLLVMDialectTranslation(*mod.getContext());
    llMod = mlir::translateModuleToLLVMIR(mod, llCtx);
    if (nullptr == llMod) {
      errStream << "\n";
      mod.print(errStream);
      errStream.flush();
      plier::report_error(llvm::Twine("Cannot generate LLVM module\n") + err);
    }
  });
  assert(nullptr != llMod);
  return serialize_mod(*llMod);
}

struct ModuleSettings {
  bool enableGpuPipeline = false;
};

void create_pipeline(plier::PipelineRegistry &registry,
                     const ModuleSettings &settings) {
  registerBasePipeline(registry);
  registerLowerToLLVMPipeline(registry);
  registerPlierToScfPipeline(registry);
  registerPlierToStdPipeline(registry);
  registerPlierToLinalgPipeline(registry);
  registerPreLowSimpleficationsPipeline(registry);
  registerParallelToTBBPipeline(registry);

  if (settings.enableGpuPipeline) {
#ifdef GPU_ENABLE
    registerLowerToGPUPipeline(registry);
#else
    plier::report_error("DPCOMP was compiled without GPU support");
#endif
  }
}

struct Module {
  mlir::MLIRContext context;
  plier::PipelineRegistry registry;
  mlir::ModuleOp module;

  Module(const ModuleSettings &settings) {
    create_pipeline(registry, settings);
  }
};

void run_compiler(Module &mod, const py::object &compilation_context) {
  auto &context = mod.context;
  auto &module = mod.module;
  auto &registry = mod.registry;

  CallbackOstream printStream;
  auto settings =
      getSettings(compilation_context["compiler_settings"], printStream);
  plier::CompilerContext compiler(context, settings, registry);
  compiler.run(module);
}
} // namespace

void init_compiler(py::dict settings) {
  auto debugType = settings["debug_type"].cast<py::list>();
  auto debugTypeSize = debugType.size();
  if (debugTypeSize != 0) {
    llvm::DebugFlag = true;
    llvm::BumpPtrAllocator alloc;
    auto types = alloc.Allocate<const char *>(debugTypeSize);
    llvm::StringSaver strSaver(alloc);
    for (size_t i = 0; i < debugTypeSize; ++i) {
      types[i] = strSaver.save(debugType[i].cast<std::string>()).data();
    }
    llvm::setCurrentDebugTypes(types, static_cast<unsigned>(debugTypeSize));
  }
}

template <typename T>
static bool getDictVal(py::dict &dict, const char *str, T &&def) {
  auto key = py::str(str);
  if (dict.contains(key))
    return dict[key].cast<T>();
  return def;
}

py::capsule create_module(py::dict settings) {
  ModuleSettings modSettings;
  modSettings.enableGpuPipeline =
      getDictVal(settings, "enable_gpu_pipeline", false);

  auto mod = std::make_unique<Module>(modSettings);
  {
    mlir::OpBuilder builder(&mod->context);
    mod->module = mlir::ModuleOp::create(builder.getUnknownLoc());
  }
  py::capsule capsule(mod.get(),
                      [](void *ptr) { delete static_cast<Module *>(ptr); });
  mod.release();
  return capsule;
}

py::capsule lower_function(const py::object &compilation_context,
                           const py::capsule &py_mod,
                           const py::object &func_ir) {
  auto mod = static_cast<Module *>(py_mod);
  auto &context = mod->context;
  auto &module = mod->module;
  auto func =
      plier_lowerer(context).lower(compilation_context, module, func_ir);
  return py::capsule(func.getOperation()); // no dtor, func owned by module
}

py::bytes compile_module(const py::object &compilation_context,
                         const py::capsule &py_mod) {
  auto mod = static_cast<Module *>(py_mod);
  run_compiler(*mod, compilation_context);
  return gen_ll_module(mod->module);
}

py::str module_str(const py::capsule &py_mod) {
  auto mod = static_cast<Module *>(py_mod);
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  mod->module.print(ss);
  ss.flush();
  return py::str(ss.str());
}
