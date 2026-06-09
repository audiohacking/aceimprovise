#pragma once

#include "aceimprovise/session/performance_session.hpp"

#include <string>

namespace aceimprovise {

std::string session_to_json(const SessionView & view);
std::string error_json(const std::string & message);

}  // namespace aceimprovise
