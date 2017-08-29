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

        inline void initialize(const Allocator &inAllocator, uint16_t inWidthOfX) {
            mStorage = inAllocator.allocateArray<double, dbal::AggregateContext,
                dbal::DoZero, dbal::ThrowBadAlloc>(arraySize(inWidthOfX));
            rebind(inWidthOfX);
            widthOfX = inWidthOfX;
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
                widthOfX != inOtherState.widthOfX)
                throw std::logic_error("Internal error: Incompatible transition "
                                       "states");

            numRows += inOtherState.numRows;
            int i;
            for (i = 0 ; i < widthOfX*widthOfX ; i ++){
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
            weight.fill(9999999);
        }

    private:
        static inline size_t arraySize(const uint16_t inWidthOfX) {
            return 3 + 2 * inWidthOfX * inWidthOfX;
        }
        void rebind(uint16_t inWidthOfX) {
            iteration.rebind(&mStorage[0]);
            widthOfX.rebind(&mStorage[1]);
            weight.rebind(&mStorage[2], inWidthOfX * inWidthOfX);
            parent.rebind(&mStorage[2 + inWidthOfX * inWidthOfX],
                inWidthOfX * inWidthOfX);
            numRows.rebind(&mStorage[2 + 2 * inWidthOfX * inWidthOfX]);
        }

        Handle mStorage;
    public:
        typename HandleTraits<Handle>::ReferenceToUInt32 iteration;
        typename HandleTraits<Handle>::ReferenceToUInt16 widthOfX;

        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap weight;
        typename HandleTraits<Handle>::ColumnVectorTransparentHandleMap parent;
        typename HandleTraits<Handle>::ReferenceToUInt64 numRows;
};

AnyType
graph_apsp_step_transition::run(AnyType &args) {
    APSPTransitionState<MutableArrayHandle<double> > state = args[0];
    if (args[1].isNull() || args[2].isNull()) { return args[0]; }

    int src = args[1].getAs<int>();
    int dest = args[2].getAs<int>();
    int weight = args[3].getAs<double>();

    int widthOfX = state.widthOfX;
    elog(WARNING,"here1");
    if (widthOfX == 0) {
        elog(WARNING,"here11");

        if (!args[4].isNull()) {
            elog(WARNING,"here12");
            APSPTransitionState<ArrayHandle<double> > previousState = args[4];

            elog(WARNING,"here13");
            state = previousState;

            elog(WARNING,"here14");
            state.reset();
        }
        elog(WARNING,"here15");
    }
    // Now do the transition step
    state.numRows++;
    elog(WARNING,"here2");
    // int i = src*state.widthOfX + dest;
    // if ( state.weight[i] > weight){
    //     state.weight[i] = weight;
    //     state.parent[i] = parent;
    // }

    int i;
    int start = src*widthOfX;
    elog(WARNING,"here3");
    for ( i = 0 ; i < widthOfX ; i ++ ){

        if (state.weight[start+i] > weight + state.weight[i*widthOfX+dest]){
            elog(WARNING,"here4");
            state.weight[start+i] = weight + state.weight[i*widthOfX+dest];
            state.parent[start+i] = dest;
        }
    }
    elog(WARNING,"here5");
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
internal_graph_apsp_result::run(AnyType &args) {
    APSPTransitionState<ArrayHandle<double> > state = args[0];
    if (state.status == NULL_EMPTY)
        return Null();

    SymmetricPositiveDefiniteEigenDecomposition<Matrix> decomposition(
        state.X_transp_AX, EigenvaluesOnly, ComputePseudoInverse);

    return stateToResult(*this, state.coef,
                         state.X_transp_AX, state.logLikelihood,
                         state.status, state.numRows);
}


} // namespace graph
} // namespace modules
} // namespace madlib
