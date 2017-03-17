#ifndef NN_Minimizer_H
#define NN_Minimizer_H

#include "simpleNN.h"
#include "simpleNN_FCN.h"
#include "simpleNN_FCN.cxx"
#include "base_plot.h"

#include "Minuit2/FunctionMinimum.h"
#include "Minuit2/MnUserParameterState.h"
#include "Minuit2/MnPrint.h"
#include "Minuit2/MnMigrad.h"
#include "Minuit2/MnMinos.h"
#include "Minuit2/MnContours.h"
#include "Minuit2/MnPlot.h"
#include "Minuit2/MinosError.h"
#include "Minuit2/ContoursError.h"
#include "Minuit2/Minuit2Minimizer.h"
#include "Math/Integrator.h"
#include "Math/Factory.h"
#include "Math/Functor.h"
#include "Math/GSLMCIntegrator.h"
#include "Math/IntegratorMultiDim.h"
#include "Math/IOptions.h"
#include "Math/IntegratorOptions.h"
#include "Math/AllIntegrationTypes.h"
#include "Math/AdaptiveIntegratorMultiDim.h"

#include <TROOT.h>
#include <TChain.h>
#include <TTree.h>
#include <TFile.h>
#include <TString.h>
#include <TMath.h>

#include <iostream>
using namespace std;
#include <ctime>

using namespace ROOT::Minuit2;

class NN_Minimizer{
	public:
		TString sFilename;
		TString bFilename;
		TString treeName;
		TString outFilename;
		TFile * fout;
		TChain * c_s;
		TChain * c_b;
		TTree * t_s_cut;
		TTree * t_b_cut;
		base * b_s;
		base * b_b;
		base * b_s_test;
		base * b_b_test;
		TTree * train_s;
		TTree * train_b;
		TTree * test_s;
		TTree * test_b;
		TTree * tout_train_s;
		TTree * tout_train_b;
		TTree * tout_test_s;
		TTree * tout_test_b;
		Long64_t nentries;
		Long64_t nentries_s;
		Long64_t nentries_b;
		Long64_t nentries_s_test;
		Long64_t nentries_b_test;
		Long64_t nentries_s_train;
		Long64_t nentries_b_train;
		vector<TString>* var_input;
		vector<TString>* spectator_var_input;
		vector<double>* var_max;
		vector<double>* var_min;
		vector<double>* var_max_int;
		vector<double>* var_min_int;

		int nEpochs;
		int nLayer;
		int nNodes[20];
		TString NeuronType;

		TString cut_s;
		TString cut_b;
		TString weightExpression;
		bool isPrintEvolution;

		vector<vector<double>*>* weight[20];
		vector<vector<double>*>* weight_init[20];
		double discriminant;

		std::vector<double> init_par;
		std::vector<double> par;
		int total;

		double duration;

		ROOT::Minuit2::Minuit2Minimizer* minimizer;

		NN_Minimizer(TString SignalFile, TString BackgroundFile, TString treename, TString outfilename);
		virtual void SetNLayer(int nlayer);
		void SetNNodes(int i, int nnodes){nNodes[i]=nnodes; cout<<"Number of nodes in inner layer "<<i<<" is "<<nNodes[i]<<endl;}
		void AddVariable(TString var){cout<<"Add input variable: "<<var<<endl; var_input->push_back(var); nNodes[0]=var_input->size();}
		void AddSpectator(TString var){cout<<"Add spectator: "<<var<<endl; spectator_var_input->push_back(var);}
		void SetNeuronType(TString myNeuronType){NeuronType=myNeuronType; cout<<"NeuronType="<<NeuronType<<endl;}
		void SetNEpochs(int N){nEpochs=N;cout<<"nEpochs = "<< N <<endl;}

		virtual void SetCut(const TString& cut, const TString& className);
		void SetWeightExpression(TString expression){weightExpression=expression; cout<<"Weight expression is: \""<< weightExpression<<"\""<<endl;}
		void IsPrintEvolution(bool b){isPrintEvolution=b; cout<<"isPrintEvolution="<<isPrintEvolution<<endl;}

		virtual void myana();

		virtual void InitTrees();
		virtual void InitParameters();
		virtual void doMinimize();
		virtual void Eval_Tree();
		virtual void plotHist();
		virtual void plotROC();
};

NN_Minimizer::NN_Minimizer(TString SignalFile, TString BackgroundFile, TString treename, TString outfilename){
	cout<<"Signal file: "<<SignalFile<<endl;
	cout<<"Background file: "<<BackgroundFile<<endl;
	sFilename = SignalFile;
	bFilename = BackgroundFile;
	treeName = treename;
	outFilename = outfilename;
	TFile * fout = new TFile(outFilename.Data(),"RECREATE");

	c_s = new TChain(treeName.Data(),"c_s");
	c_b = new TChain(treeName.Data(),"c_b");
	c_s->Add(sFilename.Data());
	c_b->Add(bFilename.Data());
	b_s = new base();
	b_b = new base();
	b_s_test = new base();
	b_b_test = new base();

	var_input = new vector<TString>;
	spectator_var_input = new vector<TString>;
	var_max = new vector<double>;
	var_min = new vector<double>;
	var_max_int = new vector<double>;
	var_min_int = new vector<double>;

	for(int i=0;i<20;i++){
		nNodes[i]=0;
		weight[i] = new vector<vector<double>*>;
		weight_init[i] = new vector<vector<double>*>;
	}
	
	NeuronType="tanh";
	nEpochs=-1;
	isPrintEvolution=false;

	minimizer = new ROOT::Minuit2::Minuit2Minimizer( ROOT::Minuit2::kMigrad );
	minimizer->SetPrintLevel(0);

	fout->cd();
	t_s_cut = new TTree();
	t_b_cut = new TTree();
	train_s = new TTree();
	train_b = new TTree();
	test_s = new TTree();
	test_b = new TTree();
	tout_train_s = new TTree();
	tout_train_b = new TTree();
	tout_test_s = new TTree();
	tout_test_b = new TTree();
}

void NN_Minimizer::SetNLayer(int nlayer){
	cout<<endl;
	nLayer=nlayer;
	if(nlayer>18){
		cout<<"Too much inner layers. Set number of inner layers = 18"<<endl;
		nLayer=18;
	}
	nNodes[nLayer+1]=1;
	cout<<"Number of inner layers = "<<nLayer<<endl;
}

void NN_Minimizer::SetCut(const TString& cut, const TString& className){
	if(className=="Signal"){
		cut_s = cut;
		cout<<"Cut on signal sample: "<<cut_s<<endl;
	}
	else if(className=="Background"){
		cut_b = cut;
		cout<<"Cut on background sample: "<<cut_b<<endl;
	}
	else{
		cout<<"Error! Cut init failed!"<<endl;
	}
}

void NN_Minimizer::InitTrees(){
	cout<<endl;
	cout<<"Init test and training trees ..."<<endl;

	c_s->SetBranchStatus("*",0);
	c_b->SetBranchStatus("*",0);
	for(unsigned int ivar=0;ivar<var_input->size();ivar++){
		TString var_name=var_input->at(ivar);
		c_s->SetBranchStatus(var_name.Data(),1);
		c_b->SetBranchStatus(var_name.Data(),1);
	}
	for(unsigned int ivar=0;ivar<spectator_var_input->size();ivar++){
		TString var_name=spectator_var_input->at(ivar);
		c_s->SetBranchStatus(var_name.Data(),1);
		c_b->SetBranchStatus(var_name.Data(),1);
	}
	c_s->SetBranchStatus(weightExpression.Data(),1);
	c_b->SetBranchStatus(weightExpression.Data(),1);

	t_s_cut = c_s->CopyTree(cut_s.Data());
	t_b_cut = c_b->CopyTree(cut_b.Data());
	nentries=0, nentries_s=0, nentries_b=0;
	nentries_s_train=0, nentries_b_train=0, nentries_s_test=0, nentries_b_test=0;
	nentries_s = t_s_cut->GetEntries();
	nentries_b = t_b_cut->GetEntries();

	train_s = t_s_cut->CloneTree(0);
	for (Long64_t jentry=0; jentry<nentries_s;jentry=jentry+2.){
		t_s_cut->GetEntry(jentry);
		train_s->Fill();
		nentries_s_train++;
	}
	train_b = t_b_cut->CloneTree(0);
	for (Long64_t jentry=0; jentry<nentries_b;jentry=jentry+2.){
		t_b_cut->GetEntry(jentry);
		train_b->Fill();
		nentries_b_train++;
	}
	test_s = t_s_cut->CloneTree(0);
	for (Long64_t jentry=1; jentry<nentries_s;jentry=jentry+2.){
		t_s_cut->GetEntry(jentry);
		test_s->Fill();
		nentries_s_test++;
	}
	test_b = c_b->CloneTree(0);
	for (Long64_t jentry=1; jentry<nentries_b;jentry=jentry+2.){
		t_b_cut->GetEntry(jentry);
		test_b->Fill();
		nentries_b_test++;
	}

	b_s->Init(train_s);
	b_b->Init(train_b);
	b_s_test->Init(test_s);
	b_b_test->Init(test_b);

	cout<<"signal training: "<<nentries_s_train<<" background training: "<<nentries_b_train<<endl;
	cout<<"signal test: "<<nentries_s_test<<" background test: "<<nentries_b_test<<endl;
}

void NN_Minimizer::InitParameters(){
	cout<<endl;
	for(unsigned int ivar=0;ivar<var_input->size();ivar++){
		cout<<"Init input variable "<<ivar<<endl;
		TString var_name=var_input->at(ivar);
		double var0Max=TMath::Max(t_s_cut->GetMaximum(var_name.Data()),t_b_cut->GetMaximum(var_name.Data()));
		double var0Min=TMath::Min(t_s_cut->GetMinimum(var_name.Data()),t_b_cut->GetMinimum(var_name.Data()));
		cout<<var_name<<":["<<var0Min<<","<<var0Max<<"]"<<endl;
		double var0Max_int= ceil(var0Max);
		double var0Min_int=floor(var0Min);
		cout<<var_name<<"_int:["<<var0Min_int<<","<<var0Max_int<<"]"<<endl;
		var_max->push_back(var0Max);
		var_min->push_back(var0Min);
		var_max_int->push_back(var0Max_int);
		var_min_int->push_back(var0Min_int);
	}

	nNodes[0]=var_input->size();
	cout<<endl;

	cout<<"Init weights:"<<endl;
	gRandom = new TRandom3();
	gRandom->SetSeed(0);
	int index=0.;
	for(int i=1;i<=nLayer+1;i++){
		cout<<"Inner layer "<<i-1<<" to "<<i<<endl;
		for(int j=0;j<=nNodes[i];j++){
			vector<double>* tmp = new vector<double>;
			vector<double>* tmp_init = new vector<double>;
			weight[i]->push_back(tmp);
			weight_init[i]->push_back(tmp_init);
			for(int k=0;k<=nNodes[i-1];k++){
				if(j==0){
					weight[i]->at(j)->push_back(0);
					weight_init[i]->at(j)->push_back(0);
				}
				else{
					weight_init[i]->at(j)->push_back(2*gRandom->Rndm()-1);
					weight[i]->at(j)->push_back(weight_init[i]->at(j)->at(k));
					init_par.push_back(weight_init[i]->at(j)->at(k));
					index++;
				}
				cout<<"w"<<j<<k<<" = "<<weight_init[i]->at(j)->at(k)<<" ";
			}
			cout<<endl;
		}
	}
	total=index;
	cout<<"total "<<total<<" parameters"<<endl;
}

void NN_Minimizer::doMinimize(){
	simpleNN_FCN * theFCN = new simpleNN_FCN(train_s, train_b, var_input, var_max_int, var_min_int, NeuronType, weightExpression, nLayer, nNodes);
	//double par_test[total];
	//for(int i=0;i<total;i++)
	//	par_test[i]=init_par[i];
	//theFCN->Eval(par_test);

	ROOT::Math::Functor* FunctorHyp = new ROOT::Math::Functor(theFCN, &simpleNN_FCN::Eval, total);;
	minimizer->SetFunction(*FunctorHyp);
	if(nEpochs>0)minimizer->SetMaxFunctionCalls(nEpochs);
	minimizer->SetPrintLevel(2);

	double step = 0.01;
	cout<<"step="<<step<<endl;

	int index=0;
	for(int i=1;i<=nLayer+1;i++){
		//cout<<"Inner layer "<<i-1<<" to "<<i<<endl;
		for(int j=1;j<=nNodes[i];j++){
			for(int k=0;k<=nNodes[i-1];k++){
				//minimizer->SetLimitedVariable(index,Form("%d", index), init_par[index], 0.01, -5, 5);
				minimizer->SetVariable(index,Form("%d", index), init_par[index], step);
				index++;
			}
		}
	}

	bool isSuccess=false;
	int nloop=0;
	cout<<endl;
	cout<<"Start minimization ..."<<endl;
	double click1 = std::clock();

	cout<<"nloop="<<nloop<<" step="<<step<<endl;
	isSuccess=minimizer->Minimize();
	
	while(!isSuccess){
		nloop++;
		step=step*0.1;
		cout<<"nloop="<<nloop<<" step="<<step<<endl;
		index=0;
		for(int i=1;i<=nLayer+1;i++){
			for(int j=1;j<=nNodes[i];j++){
				for(int k=0;k<=nNodes[i-1];k++){
					minimizer->SetVariableStepSize(index, step);
					index++;
				}
			}
		}
		minimizer->SetMaxFunctionCalls(nEpochs*nloop);
		isSuccess=minimizer->Minimize();
	}

	duration = ( std::clock() - click1 ) / (double) CLOCKS_PER_SEC;
	cout<<"duration="<<duration<<" sec"<<endl;
	cout<<endl;

	const double *xs = minimizer->X();
	cout<<"Final weights: "<<endl;
	index=0;
	for(int i=1;i<=nLayer+1;i++){
		cout<<"Inner layer "<<i-1<<" to "<<i<<endl;
		weight[i]->clear();
		for(int j=0;j<=nNodes[i];j++){
			vector<double>* tmp = new vector<double>;
			weight[i]->push_back(tmp);
			for(int k=0;k<=nNodes[i-1];k++){
				if(j==0){
					weight[i]->at(j)->push_back(0);
				}
				else{
					weight[i]->at(j)->push_back(xs[index]);
					cout<<"w"<<j<<k<<" = "<<xs[index]<<" ";
					index++;
				}
			}
			if(j!=0)	cout<<endl;
		}
	}

	cout<<"min Error = "<<theFCN->Eval(xs)<<endl;

}

void NN_Minimizer::Eval_Tree(){
	simpleNN_FCN * theFCN_train = new simpleNN_FCN(train_s, train_b, var_input, var_max_int, var_min_int, NeuronType, weightExpression, nLayer, nNodes);
	tout_train_s = train_s->CloneTree(0);
	theFCN_train->Eval_Tree(weight, train_s, tout_train_s);
	tout_train_b = train_b->CloneTree(0);
	theFCN_train->Eval_Tree(weight, train_b, tout_train_b);
	tout_test_s = test_s->CloneTree(0);
	theFCN_train->Eval_Tree(weight, test_s, tout_test_s);
	tout_test_b = test_b->CloneTree(0);
	theFCN_train->Eval_Tree(weight, test_b, tout_test_b);

	//tout_train_s->Print();
	//tout_train_b->Print();
	//tout_test_s->Print();
	//tout_test_b->Print();

	tout_train_s->Write("train_s");
	tout_train_b->Write("train_b");
	tout_test_s->Write("test_s");
	tout_test_b->Write("test_b");
}

void NN_Minimizer::plotHist(){
	gStyle->SetOptStat(0);

	TCanvas * c = new TCanvas("c","c",10,10,700,700);
	TPad * p1 = new TPad("p1","p1", 0.00,0.00,1.00,0.97);
	p1->SetFillColor(0);
	p1->Draw();
	p1->SetLeftMargin(0.14);
	p1->cd();

	TH1D * h_s_train = GetHistoWeight(tout_train_s, "discriminant", 50, -1., 1., cut_s.Data(), weightExpression.Data(), "h_s_train");
	TH1D * h_b_train = GetHistoWeight(tout_train_b, "discriminant", 50, -1., 1., cut_b.Data(), weightExpression.Data(), "h_b_train");
	TH1D * h_s_test = GetHistoWeight(tout_test_s, "discriminant", 50, -1., 1., cut_s.Data(), weightExpression.Data(), "h_s_test");
	TH1D * h_b_test = GetHistoWeight(tout_test_b, "discriminant", 50, -1., 1., cut_b.Data(), weightExpression.Data(), "h_b_test");
	h_s_test->SetLineColor(2);
	h_b_test->SetLineColor(1);

	h_s_test->SetLineColor(2);
	h_b_test->SetLineColor(1);
	h_s_test->SetMaximum((1.5*TMath::Max(h_s_test->GetMaximum(), h_b_test->GetMaximum())));
	//h_s_test->SetTitle(";Discriminant; Event / bin");
	h_s_test->SetTitle(";Discriminant; Unit Area");
	h_s_test->GetYaxis()->SetTitleOffset(1.8);
	h_s_test->Sumw2();
	h_b_test->Sumw2();
	h_s_test->DrawNormalized("HIST");
	h_b_test->DrawNormalized("HISTsame");
	h_s_train->SetMarkerColor(2);
	h_s_train->SetMarkerStyle(20);
	h_b_train->SetMarkerColor(1);
	h_b_train->SetMarkerStyle(20);
	h_s_train->Sumw2();
	h_b_train->Sumw2();
	h_s_train->DrawNormalized("PEsame");
	h_b_train->DrawNormalized("PEsame");

	TLegend* legend_test = new TLegend(0.15,0.75,0.5,0.9,"");
	legend_test->SetFillColor(kWhite);
	legend_test->AddEntry("h_s_test", "Signal (test sample)", "l");
	legend_test->AddEntry("h_b_test", "Background (test sample)", "l");
	legend_test->Draw();
	TLegend* legend_train = new TLegend(0.5,0.75,0.85,0.9,"");
	legend_train->SetFillColor(kWhite);
	legend_train->AddEntry("h_s_train", "Signal (training sample)", "p");
	legend_train->AddEntry("h_b_train", "Background (training sample)", "p");
	legend_train->Draw();

	c->Print("discriminant.png");
	h_s_test->Write("discriminant_s_test");
	h_b_test->Write("discriminant_b_test");
	h_s_train->Write("discriminant_s_train");
	h_b_train->Write("discriminant_b_train");
}

void NN_Minimizer::plotROC(){
	gStyle->SetOptStat(0);

	TCanvas * c = new TCanvas("c","c",10,10,700,700);

	TPad * p1 = new TPad("p1","p1", 0.00,0.00,1.00,0.97);
	p1->SetFillColor(0);
	p1->Draw();
	p1->SetTicks(1,1);
	p1->SetGridx();
	p1->SetGridy();
	p1->cd();
	TGraph * g_ROC_train=GetEffSvsEffB(tout_train_s, tout_train_b, cut_s.Data(), cut_b.Data(), "discriminant", -1, 1, weightExpression.Data(), 50, "g_ROC_train");
	TH2D* hGrid = new TH2D("Grid","Grid",1000,0,1,1000,0,1);
	hGrid->Draw();
	hGrid->SetTitle("training sample");
	hGrid->GetYaxis()->SetTitle("Signal Efficiency");
	hGrid->GetXaxis()->SetTitle("Background Efficiency");
	hGrid->GetYaxis()->SetTitleOffset(1.4);
	g_ROC_train->Draw("*same");
	c->SaveAs("ROC_train.png");
	g_ROC_train->Write();

	c->Clear();
	TPad * p2 = new TPad("p2","p2", 0.00,0.00,1.00,0.97);
	p2->SetFillColor(0);
	p2->Draw();
	p2->SetTicks(1,1);
	p2->SetGridx();
	p2->SetGridy();
	p2->cd();
	TGraph * g_ROC_test=GetEffSvsEffB(tout_test_s, tout_test_b, cut_s.Data(), cut_b.Data(), "discriminant", -1, 1, weightExpression.Data(), 50, "g_ROC_test");
	hGrid->SetTitle("test sample");
	hGrid->Draw();
	g_ROC_test->Draw("*same");
	c->SaveAs("ROC_test.png");
	g_ROC_test->Write();
}
#endif
