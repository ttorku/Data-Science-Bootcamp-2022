from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage
import json
import os


class WorkflowState(TypedDict):
    """State object for the LangGraph workflow"""
    chunks: List[str]
    enhanced_chunks: List[Dict[str, Any]]
    vector_results: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    final_report: str
    current_step: str
    metadata: Dict[str, Any]


class LanguageLookupWorkflow:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
        
        # Create the workflow graph
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        # Define the workflow
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("chunk_analyzer", self._analyze_chunks)
        workflow.add_node("pattern_detector", self._detect_patterns)
        workflow.add_node("match_evaluator", self._evaluate_matches)
        workflow.add_node("report_generator", self._generate_report)
        
        # Define the flow
        workflow.set_entry_point("chunk_analyzer")
        workflow.add_edge("chunk_analyzer", "pattern_detector")
        workflow.add_edge("pattern_detector", "match_evaluator")
        workflow.add_edge("match_evaluator", "report_generator")
        workflow.add_edge("report_generator", END)
        
        return workflow.compile()

    def _analyze_chunks(self, state: WorkflowState) -> WorkflowState:
        """Analyze document chunks for language-related content"""
        
        chunks = state["chunks"]
        
        analysis_prompt = PromptTemplate(
            input_variables=["chunk"],
            template="""
            Analyze this document chunk for AI/ML/Data Science language patterns:
            
            Chunk: {chunk}
            
            Identify:
            1. Technical terms and concepts
            2. Domain-specific language
            3. Key phrases that indicate AI/ML topics
            4. Confidence level (0-1) for AI/ML relevance
            
            Return your analysis as JSON with keys: technical_terms, domain, key_phrases, confidence_score
            """
        )
        
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks[:5]):  # Limit for demo
            try:
                prompt = analysis_prompt.format(chunk=chunk[:500])  # Limit chunk size
                response = self.llm.invoke([HumanMessage(content=prompt)])
                
                try:
                    analysis = json.loads(response.content)
                except json.JSONDecodeError:
                    analysis = {
                        "technical_terms": [],
                        "domain": "unknown",
                        "key_phrases": [],
                        "confidence_score": 0.5
                    }
                
                enhanced_chunk = {
                    "index": i,
                    "text": chunk,
                    "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "analysis": analysis,
                    "word_count": len(chunk.split())
                }
                enhanced_chunks.append(enhanced_chunk)
                
            except Exception as e:
                print(f"Error analyzing chunk {i}: {e}")
                enhanced_chunks.append({
                    "index": i,
                    "text": chunk,
                    "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                    "analysis": {"error": str(e)},
                    "word_count": len(chunk.split())
                })
        
        state["enhanced_chunks"] = enhanced_chunks
        state["current_step"] = "chunk_analysis_complete"
        return state

    def _detect_patterns(self, state: WorkflowState) -> WorkflowState:
        """Detect patterns across all analyzed chunks"""
        
        enhanced_chunks = state["enhanced_chunks"]
        
        # Aggregate technical terms across chunks
        all_terms = []
        domains = []
        total_confidence = 0
        
        for chunk in enhanced_chunks:
            analysis = chunk.get("analysis", {})
            if "technical_terms" in analysis:
                all_terms.extend(analysis["technical_terms"])
            if "domain" in analysis:
                domains.append(analysis["domain"])
            if "confidence_score" in analysis:
                total_confidence += analysis["confidence_score"]
        
        # Detect overall patterns
        pattern_prompt = PromptTemplate(
            input_variables=["terms", "domains"],
            template="""
            Based on these technical terms and domains from document analysis:
            
            Technical Terms: {terms}
            Domains: {domains}
            
            Identify:
            1. Main themes and topics
            2. Technology stack mentioned
            3. Application areas
            4. Overall document focus
            
            Provide insights as JSON with keys: main_themes, tech_stack, applications, document_focus
            """
        )
        
        try:
            prompt = pattern_prompt.format(
                terms=str(all_terms[:20]),  # Limit terms
                domains=str(list(set(domains)))
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                patterns = json.loads(response.content)
            except json.JSONDecodeError:
                patterns = {
                    "main_themes": ["AI/ML"],
                    "tech_stack": [],
                    "applications": [],
                    "document_focus": "Technical document"
                }
        except Exception as e:
            patterns = {"error": str(e)}
        
        state["analysis_results"] = {
            "patterns": patterns,
            "aggregated_terms": list(set(all_terms)),
            "average_confidence": total_confidence / len(enhanced_chunks) if enhanced_chunks else 0,
            "total_chunks": len(enhanced_chunks)
        }
        state["current_step"] = "pattern_detection_complete"
        return state

    def _evaluate_matches(self, state: WorkflowState) -> WorkflowState:
        """Evaluate the quality of vector search matches"""
        
        vector_results = state.get("vector_results", [])
        analysis_results = state.get("analysis_results", {})
        
        # Count matches
        found_matches = sum(1 for result in vector_results if result.get("status") == "Language found")
        total_chunks = len(vector_results)
        success_rate = (found_matches / total_chunks * 100) if total_chunks > 0 else 0
        
        # Evaluate match quality
        evaluation_prompt = PromptTemplate(
            input_variables=["success_rate", "patterns", "sample_matches"],
            template="""
            Evaluate the language matching results:
            
            Success Rate: {success_rate}%
            Document Patterns: {patterns}
            Sample Matches: {sample_matches}
            
            Provide evaluation insights:
            1. Match quality assessment
            2. Areas for improvement
            3. Confidence in results
            4. Recommendations
            
            Return as JSON with keys: quality_assessment, improvements, confidence, recommendations
            """
        )
        
        # Get sample matches for evaluation
        sample_matches = []
        for result in vector_results[:3]:
            if result.get("best_match"):
                sample_matches.append({
                    "chunk_preview": result["chunk_text"][:100],
                    "matched_language": result["best_match"]["language"],
                    "similarity_score": result["best_match"]["similarity_score"]
                })
        
        try:
            prompt = evaluation_prompt.format(
                success_rate=success_rate,
                patterns=str(analysis_results.get("patterns", {})),
                sample_matches=str(sample_matches)
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                evaluation = json.loads(response.content)
            except json.JSONDecodeError:
                evaluation = {
                    "quality_assessment": "Moderate",
                    "improvements": ["Increase chunk size", "Improve preprocessing"],
                    "confidence": 0.7,
                    "recommendations": ["Review threshold settings"]
                }
        except Exception as e:
            evaluation = {"error": str(e)}
        
        # Update analysis results
        analysis_results["evaluation"] = evaluation
        analysis_results["match_statistics"] = {
            "found_matches": found_matches,
            "total_chunks": total_chunks,
            "success_rate": success_rate
        }
        
        state["analysis_results"] = analysis_results
        state["current_step"] = "match_evaluation_complete"
        return state

    def _generate_report(self, state: WorkflowState) -> WorkflowState:
        """Generate comprehensive final report"""
        
        analysis_results = state.get("analysis_results", {})
        vector_results = state.get("vector_results", [])
        
        report_prompt = PromptTemplate(
            input_variables=["analysis", "statistics"],
            template="""
            Generate a comprehensive analysis report:
            
            Analysis Results: {analysis}
            Match Statistics: {statistics}
            
            Create a detailed report covering:
            1. Executive Summary
            2. Document Analysis Findings
            3. Language Matching Results
            4. Quality Assessment
            5. Recommendations
            
            Format as markdown report.
            """
        )
        
        try:
            prompt = report_prompt.format(
                analysis=str(analysis_results),
                statistics=f"Total matches: {len([r for r in vector_results if r.get('status') == 'Language found'])}/{len(vector_results)}"
            )
            response = self.llm.invoke([HumanMessage(content=prompt)])
            final_report = response.content
        except Exception as e:
            final_report = f"# Analysis Report\n\nError generating report: {str(e)}"
        
        state["final_report"] = final_report
        state["current_step"] = "workflow_complete"
        return state

    def run_workflow(self, chunks: List[str], vector_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run the complete LangGraph workflow"""
        
        initial_state = {
            "chunks": chunks,
            "enhanced_chunks": [],
            "vector_results": vector_results,
            "analysis_results": {},
            "final_report": "",
            "current_step": "starting",
            "metadata": {
                "total_chunks": len(chunks),
                "total_vector_results": len(vector_results)
            }
        }
        
        try:
            final_state = self.workflow.invoke(initial_state)
            return {
                "success": True,
                "final_state": final_state,
                "report": final_state.get("final_report", ""),
                "analysis": final_state.get("analysis_results", {}),
                "enhanced_chunks": final_state.get("enhanced_chunks", [])
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "report": f"# Workflow Error\n\nError running analysis workflow: {str(e)}",
                "analysis": {},
                "enhanced_chunks": []
            }