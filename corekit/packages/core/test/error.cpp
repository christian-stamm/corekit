#include "corekit/error.hpp"

#include <gtest/gtest.h>

namespace corekit::test {

    static_assert(std::is_copy_constructible_v<Error>);
    static_assert(std::is_copy_assignable_v<Error>);
    static_assert(std::is_move_constructible_v<Error>);
    static_assert(std::is_move_assignable_v<Error>);

    static_assert(std::is_base_of_v<Error, RuntimeError>);
    static_assert(std::is_base_of_v<Error, NotImplementedError>);
    static_assert(std::is_base_of_v<Error, InvalidArgumentError>);
    static_assert(std::is_base_of_v<Error, OutOfRangeError>);
    static_assert(std::is_base_of_v<Error, TimeoutError>);

    TEST(Error, DefaultConstructedErrorHasNoErrorState) {
        Error error;

        EXPECT_FALSE(error);
        EXPECT_EQ(error.type, Error::Type::NONE);
        EXPECT_EQ(error.message, "<NO ERROR>");
    }

    TEST(Error, RuntimeErrorStoresCorrectType) {
        RuntimeError error;

        EXPECT_TRUE(error);
        EXPECT_EQ(error.type, Error::Type::RUNTIME);
    }

    TEST(Error, RuntimeErrorStoresMessage) {
        constexpr auto msg = "runtime failure";

        RuntimeError error(msg);
        EXPECT_EQ(error.message, msg);
    }

    TEST(Error, NotImplementedErrorStoresMessage) {
        constexpr auto msg = "todo";

        NotImplementedError error(msg);
        EXPECT_EQ(error.message, msg);
    }

    TEST(Error, RuntimeErrorEvaluatesTrue) {
        RuntimeError error("failure");

        EXPECT_TRUE(error);
    }

    TEST(Error, CopyConstructionPreservesContents) {
        RuntimeError original("failure");
        Error        copy(original);

        EXPECT_EQ(copy.type, Error::Type::RUNTIME);
        EXPECT_EQ(copy.message, "failure");
    }

    TEST(Error, CopyAssignmentPreservesContents) {
        RuntimeError original("failure");

        Error copy;
        copy = original;

        EXPECT_EQ(copy.type, Error::Type::RUNTIME);
        EXPECT_EQ(copy.message, "failure");
    }

    TEST(Error, MoveConstructionPreservesErrorType) {
        RuntimeError original("failure");

        Error moved(std::move(original));
        EXPECT_EQ(moved.type, Error::Type::RUNTIME);
    }

    TEST(Error, MoveAssignmentPreservesErrorType) {
        RuntimeError original("failure");

        Error moved;
        moved = std::move(original);

        EXPECT_EQ(moved.type, Error::Type::RUNTIME);
    }

    TEST(Error, TracebackContainsMessage) {
        constexpr auto msg = "my failure";

        RuntimeError error(msg);

        const auto traceback = error.traceback();

        EXPECT_NE(traceback.find(msg), std::string::npos);
    }

    TEST(Error, TracebackContainsRuntimeType) {
        RuntimeError error("failure");

        const auto traceback = error.traceback();

        EXPECT_NE(traceback.find("RUNTIME"), std::string::npos);
    }

    TEST(Error, TracebackContainsNotImplementedType) {
        NotImplementedError error("todo");

        const auto traceback = error.traceback();

        EXPECT_NE(traceback.find("NOT_IMPLEMENTED"), std::string::npos);
    }

    TEST(Error, TracebackIsNotEmpty) {
        RuntimeError error("failure");

        EXPECT_FALSE(error.traceback().empty());
    }

}  // namespace corekit::test