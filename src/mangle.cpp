#include "mangle.hpp"

#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/Format.h>

#include <mlir/IR/Types.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/TypeRange.h>

#include <cctype>

namespace
{
static const constexpr auto PREFIX = "_Z";

template<unsigned Width, mlir::IntegerType::SignednessSemantics Sign, char Symbol>
bool mangle_int(llvm::raw_ostream& res, mlir::Type type)
{
    if (auto i = type.dyn_cast<mlir::IntegerType>())
    {
        if (i.getWidth() == Width && i.getSignedness() == Sign)
        {
            res << Symbol;
            return true;
        }
    }
    return false;
}

template<unsigned Width, char Symbol>
bool mangle_float(llvm::raw_ostream& res, mlir::Type type)
{
    if (auto i = type.dyn_cast<mlir::FloatType>())
    {
        if (i.getWidth() == Width)
        {
            res << Symbol;
            return true;
        }
    }
    return false;
}

void mangle_memref_impl(llvm::raw_ostream& res, mlir::MemRefType type);

bool mangle_memref(llvm::raw_ostream& res, mlir::Type type)
{
    if (auto m = type.dyn_cast<mlir::MemRefType>())
    {
        mangle_memref_impl(res, m);
        return true;
    }
    return false;
}

using type_mangler_t = bool(*)(llvm::raw_ostream&, mlir::Type);

static const constexpr type_mangler_t type_manglers[] = {
    &mangle_int<1, mlir::IntegerType::Signed, 'b'>,
    &mangle_int<1, mlir::IntegerType::Unsigned, 'b'>,
    &mangle_int<1, mlir::IntegerType::Signless, 'b'>,

    &mangle_int<8, mlir::IntegerType::Signed, 'a'>,
    &mangle_int<8, mlir::IntegerType::Unsigned, 'h'>,
    &mangle_int<8, mlir::IntegerType::Signless, 'c'>,

    &mangle_int<16, mlir::IntegerType::Signed, 's'>,
    &mangle_int<16, mlir::IntegerType::Unsigned, 't'>,
    &mangle_int<16, mlir::IntegerType::Signless, 's'>,

    &mangle_int<32, mlir::IntegerType::Signed, 'i'>,
    &mangle_int<32, mlir::IntegerType::Unsigned, 'j'>,
    &mangle_int<32, mlir::IntegerType::Signless, 'i'>,

    &mangle_int<64, mlir::IntegerType::Signed, 'x'>,
    &mangle_int<64, mlir::IntegerType::Unsigned, 'm'>,
    &mangle_int<64, mlir::IntegerType::Signless, 'x'>,

    &mangle_int<128, mlir::IntegerType::Signed, 'n'>,
    &mangle_int<128, mlir::IntegerType::Unsigned, 'o'>,
    &mangle_int<128, mlir::IntegerType::Signless, 'n'>,

    &mangle_float<32, 'f'>,
    &mangle_float<64, 'd'>,
    &mangle_float<80, 'e'>,
    &mangle_float<128, 'g'>,

    &mangle_memref,
};

bool check_type(mlir::Type type)
{
    llvm::raw_null_ostream ss;
    for (auto mangler : type_manglers)
    {
        if (mangler(ss, type))
        {
            return true;
        }
    }
    return false;
}

bool is_valid_char(char c)
{
    return (c >= 'a' && c <= 'z') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') ||
           (c == '_');
}

std::string escape_string(llvm::StringRef str)
{
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    for (auto c : str)
    {
        if (is_valid_char(c))
        {
            ss << c;
        }
        else
        {
            ss << "$" << llvm::format_hex_no_prefix(static_cast<unsigned>(c), 2);
        }
    }
    ss.flush();
    return ret;
}

template<typename F>
void mangle_ident_impl(llvm::raw_ostream& res, llvm::StringRef ident, F&& template_params)
{
    assert(!ident.empty());
    llvm::SmallVector<llvm::StringRef, 8> parts;
    ident.split(parts, '.');
    assert(!parts.empty());
    auto write_part = [&](auto part)
    {
        auto escaped = escape_string(part);
        if (std::isdigit(escaped.front()))
        {
            res << escaped.size() + 1 << '_' << escaped;
        }
        else
        {
            res << escaped.size() << escaped;
        }
    };
    if (parts.size() == 1)
    {
        write_part(parts.front());
        template_params(res);
    }
    else
    {
        res << 'N';
        for (auto& part : parts)
        {
            write_part(part);
        }
        template_params(res);
        res << 'E';
    }
}

void mangle_ident(llvm::raw_ostream& res, llvm::StringRef ident)
{
    auto dummy = [](auto&) {};
    mangle_ident_impl(res, ident, dummy);
}

template<typename F>
void mangle_ident(llvm::raw_ostream& res, llvm::StringRef ident, F&& template_params)
{
    auto wrap_template = [&](llvm::raw_ostream& s)
    {
        s << 'I';
        template_params(s);
        s << 'E';
    };
    mangle_ident_impl(res, ident, wrap_template);
}

void mangle_type(llvm::raw_ostream& res, mlir::Type type)
{
    for(auto m : type_manglers)
    {
        if (m(res, type))
        {
            return;
        }
    }
    llvm_unreachable("Cannot mangle type");
}

void mangle_memref_impl(llvm::raw_ostream& res, mlir::MemRefType type)
{
    auto params = [&](llvm::raw_ostream& s)
    {
        mangle_type(s, type.getElementType());
        s << "Li"<< type.getRank() << "E";
        mangle_ident(s, "C");
    };
    mangle_ident(res, "array", params);
}

void mangle_types(llvm::raw_ostream& res, mlir::TypeRange types)
{
    for (auto type : types)
    {
        mangle_type(res, type);
    }
}

}

bool mangle(llvm::raw_ostream& res, llvm::StringRef ident, mlir::TypeRange types)
{
    for (auto type : types)
    {
        if (!check_type(type))
        {
            return false;
        }
    }
    res << PREFIX;
    mangle_ident(res, ident);
    mangle_types(res, types);
    return true;
}


std::string mangle(llvm::StringRef ident, mlir::TypeRange types)
{
    std::string ret;
    llvm::raw_string_ostream ss(ret);
    if (!mangle(ss, ident, types))
    {
        return {};
    }
    ss.flush();
    return ret;
}
