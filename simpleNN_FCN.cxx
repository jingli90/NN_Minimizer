#include "simpleNN.h"
#include "simpleNN_FCN.h"
#include "base.h"
#include <cassert>

double simpleNN_FCN::Eval(const double* par) const {
	//assert(nLayer <= 18);
	unsigned int total=0;
	vector<vector<double>*>* weight[20];
	for(int i=0;i<20;i++){
		weight[i] = new vector<vector<double>*>;
		if(i>=1)
			total=total+nNodes[i]*nNodes[i-1];
	}

	int index=0;
	for(int i=1;i<=nLayer+1;i++){
		//cout<<"Inner layer "<<i-1<<" to "<<i<<endl;
		for(int j=0;j<=nNodes[i];j++){
			vector<double>* tmp = new vector<double>;
			weight[i]->push_back(tmp);
			for(int k=0;k<=nNodes[i-1];k++){
				//weight[i]->at(j)->push_back(myweight[i]->at(j)->at(k));
				if(j==0)
					weight[i]->at(j)->push_back(0);
				else{
					weight[i]->at(j)->push_back(par[index]);
					index++;
				}
				//if(j!=0)
					//cout<<"w"<<j<<k<<" = "<<weight[i]->at(j)->at(k)<<" ";
			}
			//cout<<endl;
		}
	}

	int mynNodes[20];
	for(int i=0;i<=nLayer;i++){
		mynNodes[i]=nNodes[i];
	}
	mynNodes[nLayer+1]=1;


	simpleNN myNN(weight, NeuronType, nLayer, mynNodes);

	base * b_s;
	base * b_b;
	b_s = new base();
	b_b = new base();
	b_s->Init(t_s);
	b_b->Init(t_b);
	unsigned int nentries_s=t_s->GetEntries();
	unsigned int nentries_b=t_b->GetEntries();
	unsigned int nEvents = nentries_s+nentries_b;
	Int_t* evt_index = new Int_t[nEvents];
	int isSig;
	for (unsigned int i = 0; i < nEvents; i++) evt_index[i] = i;
	//Shuffle(evt_index, nEvents);
	vector<double> * pos = new vector<double>;

	double chi2=0;
	double evt_weight=1;
	double sum_weight=0.;
	for(Long64_t jentry=0; jentry<nEvents;jentry++){
		unsigned int jentry_tmp=evt_index[jentry];
		pos->clear();
		if(jentry_tmp<nentries_s){
			isSig=1;
			b_s->GetEntry(jentry_tmp);
			for(unsigned int i=0; i<var_name->size(); i++){
				double var_tmp = b_s->GetVal(var_name->at(i).Data());
				var_tmp = 2 * (var_tmp - var_min_int->at(i)) / (var_max_int->at(i) - var_min_int->at(i)) - 1;
				pos->push_back(var_tmp);
			}
			evt_weight=b_s->GetVal(weightExpression);
		}
		else{
			//isSig=-1;
			isSig=0; // test for TMVA
			b_b->GetEntry(jentry_tmp-nentries_s);
			for(unsigned int i=0; i<var_name->size(); i++){
				double var_tmp = b_b->GetVal(var_name->at(i).Data());
				var_tmp = 2 * (var_tmp - var_min_int->at(i)) / (var_max_int->at(i) - var_min_int->at(i)) - 1;
				pos->push_back(var_tmp);
			}
			evt_weight=b_b->GetVal(weightExpression);
		}
		sum_weight=sum_weight+evt_weight;
		chi2 = chi2 + (myNN(pos)-isSig) * (myNN(pos)-isSig) * evt_weight;
		if(jentry%1000==0){
			//cout<<"jentry="<<jentry<<endl;
			//for(unsigned int i=0; i<var_name->size(); i++){
			//	cout<<var_name->at(i)<<"="<<pos->at(i)<<" ";
			//}
			//cout<<endl;
			//cout<<"isSig="<<isSig<<" ";
			//cout<<"myNN(pos)="<<myNN(pos)<<" ";
			//cout<<"evt_weight="<<evt_weight<<" ";
			//cout<<"(myNN(pos)-isSig)*(myNN(pos)-isSig)*evt_weight="<<(myNN(pos)-isSig) * (myNN(pos)-isSig) * evt_weight<<" ";
			//cout<<"chi2="<<chi2<<" ";
			//cout<<endl;
		}
	}
	chi2=chi2/sum_weight;

	cout<<"chi2="<<chi2<<endl;
	return chi2;
}

void simpleNN_FCN::Shuffle(Int_t* index, Int_t n){
	Int_t j, k;
	Int_t a = n - 1;
	gRandom = new TRandom3();
	gRandom->SetSeed(0);
	for (Int_t i = 0; i < n; i++) {
		j = (Int_t) (gRandom->Rndm() * a);
		k = index[j];
		index[j] = index[i];
		index[i] = k;
	}
}

void simpleNN_FCN::Eval_Tree(vector<vector<double>*>* weight[20], TTree * tin, TTree * tout){
	int mynNodes[20];
	for(int i=0;i<=nLayer;i++){
		mynNodes[i]=nNodes[i];
	}
	mynNodes[nLayer+1]=1;
	simpleNN myNN(weight, NeuronType, nLayer, mynNodes);

	double discriminant;
	tout->Branch("discriminant",&discriminant,"discriminant/D");
	base * b;
	b = new base();
	b->Init(tin);
	unsigned int nentries=tin->GetEntries();
	vector<double> * pos = new vector<double>;
	for(Long64_t jentry=0; jentry<nentries;jentry++){
		pos->clear();
		b->GetEntry(jentry);
		for(unsigned int i=0; i<var_name->size(); i++){
			double var_tmp = b->GetVal(var_name->at(i).Data());
			var_tmp = 2 * (var_tmp - var_min_int->at(i)) / (var_max_int->at(i) - var_min_int->at(i)) - 1;
			pos->push_back(var_tmp);
		}
		discriminant=myNN(pos);
		tout->Fill();
	}
}

