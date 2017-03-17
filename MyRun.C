void MyRun(){
	TString sFilename = "/afs/cern.ch/work/j/jing/public/IPHCNtuple/fromXavier/trees_from_Xavier_20170302/ttHToNonbb.root";
	TString bFilename = "/afs/cern.ch/work/j/jing/public/IPHCNtuple/fromXavier/trees_from_Xavier_20170302/ttV.root";
	TString treeName = "Tree";
	TString outFilename = "output.root";
	gROOT->ProcessLine(".L NN_Minimizer.C++");
	gROOT->ProcessLine(Form("NN_Minimizer s(\"%s\",\"%s\",\"%s\",\"%s\")",sFilename.Data(),bFilename.Data(),treeName.Data(),outFilename.Data()));

	gROOT->ProcessLine(Form("s.AddVariable(\"nJet25_Recl\")"));
	gROOT->ProcessLine(Form("s.AddVariable(\"max_Lep_eta\")"));
	gROOT->ProcessLine(Form("s.AddVariable(\"MT_met_lep1\")"));
	gROOT->ProcessLine(Form("s.AddVariable(\"mindr_lep1_jet\")"));
	gROOT->ProcessLine(Form("s.AddVariable(\"mindr_lep2_jet\")"));
	gROOT->ProcessLine(Form("s.AddVariable(\"LepGood_conePt0\")"));
	gROOT->ProcessLine(Form("s.AddVariable(\"LepGood_conePt1\")"));
	gROOT->ProcessLine(Form("s.AddSpectator(\"is_3l_TTH_SR\")"));
	gROOT->ProcessLine(Form("s.AddSpectator(\"mc_ttZhypAllowed\")")); //
	gROOT->ProcessLine(Form("s.SetNLayer(1)")); //
	gROOT->ProcessLine(Form("s.SetNNodes(1,8)")); //
	gROOT->ProcessLine(Form("s.SetNEpochs(500)")); //
	gROOT->ProcessLine(Form("s.SetCut(\"is_3l_TTH_SR==1 && mc_ttZhypAllowed==1\", \"Signal\")")); //
	gROOT->ProcessLine(Form("s.SetCut(\"is_3l_TTH_SR==1 && mc_ttZhypAllowed==1\", \"Background\")")); //
	gROOT->ProcessLine(Form("s.SetWeightExpression(\"weight\")"));
	gROOT->ProcessLine(Form("s.IsPrintEvolution(false)")); //
	gROOT->ProcessLine(Form("s.SetNeuronType(\"tanh\")")); //
	//gROOT->ProcessLine(Form("s.SetNeuronType(\"tanh_num\")"));
	//gROOT->ProcessLine(Form("s.SetNeuronType(\"sigmoid\")"));
	gROOT->ProcessLine("s.myana()");
	
}
