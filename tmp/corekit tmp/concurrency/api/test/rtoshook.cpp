extern "C" {

void vApplicationMallocFailedHook(void) {
    // Fail loudly during tests.
    __builtin_trap();
}

void vApplicationStackOverflowHook(void*, char*) {
    // Fail loudly during tests.
    __builtin_trap();
}

void vApplicationDaemonTaskStartupHook(void) {
    // Nothing required for tests.
}
}