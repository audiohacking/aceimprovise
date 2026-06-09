cmake_minimum_required(VERSION 3.14)
project(aceimprovise LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ACEIMPROVISE_BUILD_WEB "Build web UI before embedding (requires npm)" OFF)

find_package(Threads REQUIRED)

# Upstream inference stack (read-only submodule).
# EXCLUDE_FROM_ALL skips acestep.cpp test targets that fail in embedded builds.
add_subdirectory(third_party/acestep.cpp EXCLUDE_FROM_ALL)

add_library(aceimprovise-core STATIC
    src/aceimprovise/liveset/liveset.cpp
    src/aceimprovise/liveset/compositor.cpp
    src/aceimprovise/stream/shared_state.cpp
    src/aceimprovise/stream/pipeline.cpp
    src/aceimprovise/session/performance_session.cpp
    src/aceimprovise/inference/dit_ring_stepper.cpp
    src/aceimprovise/inference/stream_inference.cpp
    src/aceimprovise/server/protocol.cpp
    src/aceimprovise/server/http_server.cpp
)

target_include_directories(aceimprovise-core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acestep.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acestep.cpp/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acestep.cpp/tools
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acestep.cpp/vendor/yyjson
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acestep.cpp/vendor/cpp-httplib
)
target_include_directories(aceimprovise-core SYSTEM PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/acestep.cpp/ggml/include
)

target_link_libraries(aceimprovise-core PUBLIC
    acestep-core
    yyjson
    httplib
    Threads::Threads
)

if(MSVC)
    target_compile_options(aceimprovise-core PRIVATE /W4 /wd4100)
else()
    target_compile_options(aceimprovise-core PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

add_executable(ace-improvise-server tools/ace-improvise-server.cpp)
target_link_libraries(ace-improvise-server PRIVATE aceimprovise-core)

if(MSVC)
    target_compile_options(ace-improvise-server PRIVATE /W4 /wd4100)
else()
    target_compile_options(ace-improvise-server PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

if(ACEIMPROVISE_BUILD_WEB)
    add_custom_target(aceimprovise-web ALL
        COMMAND npm ci
        COMMAND npm run build
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/web
        COMMENT "Building aceimprovise web UI"
    )
    add_dependencies(ace-improvise-server aceimprovise-web)
endif()
