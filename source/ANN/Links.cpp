/*
 *  Layers.cpp
 *  rl
 *
 *  Created by Guido Novati on 11.02.16.
 *  Copyright 2016 ETH Zurich. All rights reserved.
 *
 */

#pragma once
#include "Layers.h"
#include <iomanip>
#include <cassert>

using namespace ErrorHandling;

void WhiteningLink::propagate(const Activation* const netFrom, Activation* const netTo,
																	const Real* const weights) const


void WhiteningLink::backPropagate(Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, Real* const gradW) const

void Link::propagate(const Activation* const netFrom, Activation* const netTo,
																	const Real* const weights) const


void Link::backPropagate(Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, Real* const gradW) const


void LinkToLSTM::propagate(const Activation* const netFrom, Activation* const netTo,
																	const Real* const weights) const


void LinkToLSTM::backPropagate(Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, Real* const gradW) const


void LinkToConv2D::propagate(const Activation* const netFrom, Activation* const netTo,
																	const Real* const weights) const


void LinkToConv2D::backPropagate(Activation* const netFrom, const Activation* const netTo,
											const Real* const weights, Real* const gradW) const


void Link::orthogonalize(const int n0, Real* const _weights, int nOut, int nIn) const


void Link::print() const


void LinkToLSTM::print() const


void Link::initialize(mt19937* const gen, Real* const _weights) const


void LinkToLSTM::initialize(mt19937* const gen, Real* const _weights) const


void LinkToConv2D::initialize(mt19937* const gen, Real* const _weights) const


void WhiteningLink::initialize(mt19937* const gen, Real* const _weights) const


void Link::save(std::ostringstream & o, Real* const _weights) const


void LinkToLSTM::save(std::ostringstream & o, Real* const _weights) const


void LinkToConv2D::save(std::ostringstream & o, Real* const _weights) const


void WhiteningLink::save(std::ostringstream & o, Real* const _weights) const


void Link::restart(std::istringstream & buf, Real* const _weights) const


void LinkToLSTM::restart(std::istringstream & buf, Real* const _weights) const


void LinkToConv2D::restart(std::istringstream & buf, Real* const _weights) const


void WhiteningLink::restart(std::istringstream & buf, Real* const _weights) const


void Graph::initializeWeights(mt19937* const gen, Real* const _weights, Real* const _biases) const

