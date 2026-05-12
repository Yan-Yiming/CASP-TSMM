#include "tsmm.hpp"

#include <vector>

std::vector<ImplDesc>& tsmm_impl_registry() {
    static std::vector<ImplDesc> impls;
    return impls;
}

