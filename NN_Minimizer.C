#ifndef NN_Minimizer_cxx
#define NN_Minimizer_cxx

#include "NN_Minimizer.h"

void NN_Minimizer::myana(){
	InitTrees();
	InitParameters();
	doMinimize();
	Eval_Tree();
	plotHist();
	plotROC();
}

#endif
