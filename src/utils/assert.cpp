#if defined(__linux__)
#    include <cxxabi.h>
#    include <dlfcn.h>
#    include <execinfo.h>
#endif
#include <array>
#include <cstdint>
#include <cstdio>
#include <format>
#include <memory>
#include <sstream>
#include <string>

#include "corekit/core.hpp"

namespace corekit {
    namespace utils {

        namespace {

#if defined(__linux__)
            std::string demangle_symbol(const char* name) {
                int   status = 0;
                char* demanged =
                    abi::__cxa_demangle(name, nullptr, nullptr, &status);
                std::unique_ptr<char, decltype(&std::free)> holder(demanged,
                                                                   &std::free);
                if (status == 0 && holder) {
                    return std::string(holder.get());
                }
                return std::string(name);
            }

            std::string addr2line_resolve(const void* addr,
                                          const char* module_path) {
                if (module_path == nullptr || module_path[0] == '\0') {
                    return "";
                }

                std::ostringstream cmd;
                cmd << "addr2line -e \"" << module_path << "\" -f -C -p -i "
                    << std::hex << reinterpret_cast<std::uintptr_t>(addr);

                auto deleter = [](FILE* f) {
                    pclose(f);
                };

                std::unique_ptr<FILE, decltype(deleter)> pipe(
                    popen(cmd.str().c_str(), "r"),
                    deleter);
                if (!pipe) {
                    return "";
                }

                std::array<char, 256> buffer{};
                std::string           output;
                while (fgets(buffer.data(),
                             static_cast<int>(buffer.size()),
                             pipe.get()) != nullptr) {
                    output += buffer.data();
                }

                while (!output.empty() &&
                       (output.back() == '\n' || output.back() == '\r')) {
                    output.pop_back();
                }

                for (char& ch : output) {
                    if (ch == '\n' || ch == '\r') {
                        ch = ' ';
                    }
                }

                return output;
            }

            std::string describe_frame(void* addr) {
                Dl_info    info{};
                const bool found = dladdr(addr, &info) != 0;

                std::string symbol;
                if (found && info.dli_sname) {
                    symbol = demangle_symbol(info.dli_sname);
                }

                const char* module_path = found ? info.dli_fname : nullptr;
                const auto  base        = found ? info.dli_fbase : nullptr;
                const auto  offset_addr =
                    base ? reinterpret_cast<void*>(
                               reinterpret_cast<std::uintptr_t>(addr) -
                               reinterpret_cast<std::uintptr_t>(base))
                          : addr;

                const std::string resolved =
                    addr2line_resolve(offset_addr, module_path);

                if (!resolved.empty()) {
                    return resolved;
                }

                std::ostringstream out;
                if (!symbol.empty()) {
                    out << symbol;
                } else if (found && info.dli_fname) {
                    out << info.dli_fname;
                } else {
                    out << "<unknown>";
                }

                return out.str();
            }
#endif

            std::string build_stacktrace(std::size_t skip_frames) {
#if defined(__linux__)
                constexpr int max_frames = 64;
                void*         frames[max_frames];
                const int     count   = ::backtrace(frames, max_frames);
                char**        symbols = ::backtrace_symbols(frames, count);

                if (symbols == nullptr) {
                    return "\t<stacktrace unavailable>\n";
                }

                std::ostringstream out;
                for (int i = static_cast<int>(skip_frames); i < count; ++i) {
                    const std::string described = describe_frame(frames[i]);
                    const std::string line =
                        described.empty() ? std::string(symbols[i]) : described;

                    out << "\t" << (i - static_cast<int>(skip_frames)) << ": "
                        << line << "\n";
                }

                std::free(symbols);
                return out.str();
#else
                (void)skip_frames;
                return "\t<stacktrace unavailable>\n";
#endif
            }

        }  // namespace

        void corecheck(bool               condition,
                       const std::string& message,
                       const Location&    location) {
#ifndef NDEBUG
            if (!condition) {
                const std::string trace = build_stacktrace(2);
                throw std::runtime_error(
                    std::format("Assertion failed:\n\n\tFile: {}\n\tFunc: "
                                "{}\n\tLine: {} ({})\n\tDesc: {}\n\n"
                                "\tStack:\n{}",
                                location.file_name(),
                                location.function_name(),
                                location.line(),
                                location.column(),
                                message,
                                trace));
            }
#endif
        }

        void glCheckError(std::string meta, const Location& location) {
            const GLenum& errcode = glGetError();

            corecheck(errcode == GL_NO_ERROR,
                      std::format("OpenGL Code: 0x{:04X} ({:08}) {}",
                                  errcode,
                                  errcode,
                                  meta),
                      location);
        }

    };  // namespace utils
};  // namespace corekit
