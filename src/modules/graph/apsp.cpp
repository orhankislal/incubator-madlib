/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the WARNING file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
/* ----------------------------------------------------------------------- *//**
 *
 * @file apsp.cpp
 *
 *//* ----------------------------------------------------------------------- */

#include <limits>
#include <dbconnector/dbconnector.hpp>
#include <modules/shared/HandleTraits.hpp>
#include <modules/prob/boost.hpp>
#include <boost/math/distributions.hpp>
#include <modules/prob/student.hpp>
#include "apsp.hpp"


namespace madlib {

// Use Eigen
using namespace dbal::eigen_integration;

namespace modules {

// Import names from other MADlib modules
using dbal::NoSolutionFoundException;

namespace graph {

// FIXME this enum should be accessed by all modules that may need grouping
// valid status values
enum { IN_PROCESS, COMPLETED, TERMINATED, NULL_EMPTY };

template <class Handle>
class APSPTransitionState {
    template <class OtherHandle>
    friend class APSPTransitionState;

    public:
        APSPTransitionState(const AnyType &inArray)
            : mStorage(inArray.getAs<Handle>()) {

            rebind(static_cast<uint16_t>(mStorage[1]));
        }
        inline operator AnyType() const {
            return mStorage;
        }

        inline void initialize(const Allocator &inAllocator, uint16_t inVcnt) {
            mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inVcnt));
            rebind(inVcnt);
            vcnt = inVcnt;
            weight.fill(9999999);
            parent.fill(-1);
            int i;
            for (i = 0; i < vcnt; i ++){
                weight[i*vcnt+i] = 0;
                parent[i*vcnt+i] = i;
            }
        }

        template <class OtherHandle>
        APSPTransitionState &operator=(
            const APSPTransitionState<OtherHandle> &inOtherState) {

            for (size_t i = 0; i < mStorage.size(); i++)
                mStorage[i] = inOtherState.mStorage[i];
            return *this;
        }

        template <class OtherHandle>
        APSPTransitionState &operator+=(
            const APSPTransitionState<OtherHandle> &inOtherState) {

            if (mStorage.size() != inOtherState.mStorage.size() ||
                vcnt != inOtherState.vcnt)
                throw std::logic_error("Internal error: Incompatible transition "
                                       "states");

            numRows += inOtherState.numRows;
            int i;
            for (i = 0 ; i < vcnt*vcnt ; i ++){
                if (weight[i] > inOtherState.weight[i]){
                    weight[i] = inOtherState.weight[i];
                    parent[i] = inOtherState.parent[i];
                }
            }
            return *this;
        }

        // FIXME
        inline void reset() {
            numRows = 0;
        }

    private:
        static inline size_t arraySize(const uint16_t inVcnt) {
            return 3 + 2 * inVcnt * inVcnt;
        }
        void rebind(uint16_t inVcnt) {
            iteration.rebind(&mStorage[0]);
            vcnt.rebind(&mStorage[1]);
            weight.rebind(&mStorage[2], inVcnt * inVcnt);
            parent.rebind(&mStorage[2 + inVcnt * inVcnt],
                inVcnt * inVcnt);
            numRows.rebind(&mStorage[2 + 2 * inVcnt * inVcnt]);
        }

        Handle mStorage;
    public:
        typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
        typename HandleTraits<Handle>::ReferenceToUInt16 vcnt;

        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap weight;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap parent;
        typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
};

AnyType
graph_apsp_step_transition::run(AnyType &args) {
    APSPTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }

    int vcnt = args[1].getAs<int>();
    int src = args[2].getAs<int>();
    int dest = args[3].getAs<int>();
    double weight = args[4].getAs<double>();
    if (state.numRows == 0) {

        state.initialize(*this, static_cast<uint16_t>(vcnt));
        if (!args[5].isNull()) {
            APSPTransitionState<ArrayHandle<double> > previousState = args[5];
            state = previousState;
            state.reset();
        }
    }
    // Now do the transition step
    state.numRows++;

    int i;
    int start = src*vcnt;

    for ( i = 0 ; i < vcnt ; i ++ ){

        if (state.weight[dest*vcnt+i] != 9999999 && state.weight[start+i] > weight + state.weight[dest*vcnt+i]){
            // elog(WARNING,"comp i = %d, %d %d, %f > %f + %f", i, src, dest,
            //    state.weight[start+i], weight, state.weight[dest*vcnt+i]);
            state.weight[start+i] = weight + state.weight[dest*vcnt+i];
            state.parent[start+i] = dest;
        }
    }
    return state;
}

AnyType
graph_apsp_step_merge_states::run(AnyType &args) {
    APSPTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    APSPTransitionState<ArrayHandle<double> > stateRight = args[1];

    // We first handle the trivial case where this function is called with one
    // of the states being the initial state
    if (stateLeft.numRows == 0)
        return stateRight;
    else if (stateRight.numRows == 0)
        return stateLeft;

    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}

AnyType
graph_apsp_step_final::run(AnyType &args) {
    APSPTransitionState<MutableArrayHandle<double> > state = args[0];

    return state;
}


AnyType
internal_graph_apsp_finalizer::run(AnyType &args) {

    //elog(WARNING, "here1");
    APSPTransitionState<ArrayHandle<double> > state = args[0];
    if (state.numRows == 0)
        return Null();

    int vcnt = state.vcnt;
    MutableNativeIntegerVector src(allocateArray<int>(vcnt*vcnt));
    MutableNativeIntegerVector dest(allocateArray<int>(vcnt*vcnt));
    MutableNativeColumnVector weight(allocateArray<double>(vcnt*vcnt));
    MutableNativeIntegerVector parent(allocateArray<int>(vcnt*vcnt));
    int i,j;
    for (i = 0 ; i < vcnt ; i ++){
        for (j = 0 ; j < vcnt ; j ++){
            src[i*vcnt+j] = i;
            dest[i*vcnt+j] = j;
            weight[i*vcnt+j] = state.weight[i*vcnt+j];
            parent[i*vcnt+j] = state.parent[i*vcnt+j];
        }
    }

    AnyType tuple;

    tuple << src << dest << weight << parent;
    return tuple;
}


} // namespace graph
} // namespace modules
} // namespace madlib
