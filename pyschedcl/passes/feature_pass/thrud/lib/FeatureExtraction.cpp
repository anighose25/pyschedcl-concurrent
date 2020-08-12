#include "thrud/FeatureExtraction.h"

#include "thrud/DivergenceAnalysis.h"
#include "thrud/Utils.h"

#include "llvm/Analysis/ScalarEvolution.h"

cl::opt<std::string> kernelName("count-kernel-name", cl::init(""), cl::Hidden,
                                cl::desc("Name of the kernel to analyze"));

extern cl::opt<unsigned int> CoarseningDirectionCL;

char OpenCLFeatureExtractor::ID = 0;
static RegisterPass<OpenCLFeatureExtractor> X("opencl-instcount",
                                              "Collect opencl features");

//------------------------------------------------------------------------------
bool OpenCLFeatureExtractor::runOnFunction(Function &function) {
 // errs() << function.getName() <<"\n";
 // errs() << kernelName <<"\n";
  if (function.getName() != kernelName){
  //  errs() << "CL filename != kernel name!" <<"\n";
    return false;
  }
  pdt = &getAnalysis<PostDominatorTree>();
  dt = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  mdda = &getAnalysis<MultiDimDivAnalysis>();
  sdda = &getAnalysis<SingleDimDivAnalysis>();
  se = &getAnalysis<ScalarEvolution>();
  ndr = &getAnalysis<NDRange>();
  loopInfo = &getAnalysis<LoopInfo>();
    
  collector.getAccessExpressions(function, ndr,loopInfo,pdt);
  collector.getFunctionTids(function, ndr);
  // ndr->dump(); 
  visit(function);

  //errs()<<"Dumping statistics\n";
  collector.dump();
  //outs()<<"Test "<<"\n";
  return false;
}

//------------------------------------------------------------------------------
void OpenCLFeatureExtractor::getAnalysisUsage(AnalysisUsage &au) const {
  au.addRequired<LoopInfo>();
  au.addPreserved<LoopInfo>();
  au.addRequired<MultiDimDivAnalysis>();
  au.addRequired<SingleDimDivAnalysis>();
  au.addRequired<PostDominatorTree>();
  au.addRequired<DominatorTreeWrapperPass>();
  au.addRequired<ScalarEvolution>();
  au.setPreservesAll();
  au.addRequired<NDRange>();
}

//------------------------------------------------------------------------------
// Count all instruction types.
#define HANDLE_INST(N, OPCODE, CLASS)                                          \
  void OpenCLFeatureExtractor::visit##OPCODE(CLASS &) {                        \
    collector.instTypes[#OPCODE] += 1;                                         \
    collector.instTypes["insts"] += 1;                                         \
  }
#include "llvm/IR/Instruction.def"

//------------------------------------------------------------------------------
void OpenCLFeatureExtractor::visitInstruction(Instruction &inst) {
  errs() << "Unknown instruction: " << inst;
  llvm_unreachable(0);
}

//------------------------------------------------------------------------------
void OpenCLFeatureExtractor::visitBasicBlock(BasicBlock &basicBlock) {
//  BasicBlock *block = (BasicBlock *)&basicBlock;
  collector.instTypes["blocks"] += 1;
  //collector.computeILP(block);
  //collector.computeMLP(block);
  collector.countInstsBlock(basicBlock);
  collector.countConstants(basicBlock);
  collector.countBarriers(basicBlock);
  collector.countMathFunctions(basicBlock);
  collector.countOutgoingEdges(basicBlock);
  collector.countIncomingEdges(basicBlock);
  collector.countLocalMemoryUsage(basicBlock);
  collector.countPhis(basicBlock);
  collector.livenessAnalysis(basicBlock);
}

//------------------------------------------------------------------------------
void OpenCLFeatureExtractor::visitFunction(Function &function) {
  // Extract ThreadId values. 
  collector.getFunctionSignature(function);
  collector.countBranches(function);
  collector.countEdges(function);
  collector.countDivInsts(function, mdda, sdda);
  collector.countArgs(function);
}
