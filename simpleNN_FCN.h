#ifndef simpleNN_FCN_H
#define simpleNN_FCN_H

#include <vector>

#include "TString.h"
#include "TTree.h"


class simpleNN_FCN{
	public:
		simpleNN_FCN(TTree * my_t_s,
				TTree * my_t_b,
				vector<TString> * myvar_name,
				vector<double>* var_max,
				vector<double>* var_min,
				TString myNeuronType, 
				TString weight,
				int mynLayer, 
				int mynNodes[20]);

		~simpleNN_FCN() {}

		virtual double Up() const {return theErrorDef;}
		virtual double Eval(const double *) const;
		virtual void Shuffle(Int_t* index, Int_t n); 

		void setErrorDef(double def) {theErrorDef = def;}

		virtual void Eval_Tree(vector<vector<double>*>* weight[20], TTree * tin, TTree * tout);

	private:
		double theErrorDef;
		TString NeuronType;
		int nLayer;
		int nNodes[20];
		TTree * t_s;
		TTree * t_b;
		vector<TString> * var_name;
		vector<double>* var_max_int;
		vector<double>* var_min_int;
		TString weightExpression;
};

simpleNN_FCN::simpleNN_FCN(TTree * my_t_s,
		TTree * my_t_b,
		vector<TString> * myvar_name,
		vector<double>* var_max,
		vector<double>* var_min,
		TString myNeuronType,
		TString weight,
		int mynLayer, 
		int mynNodes[20]){
	theErrorDef=1.;

	nLayer=mynLayer;
	NeuronType=myNeuronType;
	for(int i=0;i<20;i++){
		nNodes[i]=0;
	}
	for(int i=0;i<=mynLayer;i++){
		nNodes[i]=mynNodes[i];
	}
	nNodes[mynLayer+1]=1;

	t_s=my_t_s;
	t_b=my_t_b;

	var_name=new vector<TString>;
	var_max_int = new vector<double>;
	var_min_int = new vector<double>;
	for(unsigned int i=0; i<myvar_name->size(); i++){
		var_name->push_back(myvar_name->at(i));
		var_max_int->push_back(var_max->at(i));
		var_min_int->push_back(var_min->at(i));
	}
	nNodes[0]=var_name->size();

	weightExpression=weight;
}



#endif
