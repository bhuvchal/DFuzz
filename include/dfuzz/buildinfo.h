#pragma once
namespace dfuzz {
// A conformance claim without its build configuration is unfalsifiable.
// Printed at startup by every harness binary and emitted into result records.
void print_build_config();
}
