# Project Sahayak - Educational Content Generation System

## System Overview

Project Sahayak is an AI-powered educational content generation system that creates personalized lesson plans and quizzes. The system is built using a directed graph architecture where each node is a specialized component that handles a specific aspect of content generation.

### High-Level Flow

[First Diagram Here - Sequence Diagram]

This diagram shows the high-level interaction between system layers:
1. User provides input (voice/text)
2. System processes and extracts intent
3. RAG system searches for relevant content
4. Parallel content generation occurs
5. Content is verified and stored

### Node Architecture

[Second Diagram Here - Node Flow Diagram]

The system is composed of interconnected nodes:
1. **Entry Point**: Intent Parser
2. **Knowledge Processing**: RAG Agent
3. **Content Generation**: Parallel processing of lesson, quiz, and images
4. **Quality Control**: Hallucination Guard and Evaluation
5. **Persistence**: Memory Agent

### State Management

[Third Diagram Here - State Management Diagram]

The state flows through the system with each node updating specific parts:
1. Initial state contains user request
2. RAG adds source content
3. Generators add their content
4. Verification adds checks
5. Final state includes all content and evaluations

## Solution Structure

### Core Components

1. **Graph State**
   - Central state management using TypedDict
   - Tracks entire system state
   - Ensures type safety
   - Key sections:
     ```python
     class GraphState(TypedDict):
         # Input fields (user interaction)
         user_request: str
         topic: str
         grade_level: str
         
         # Knowledge fields (RAG system)
         retriever: object
         grounded_content: str
         
         # Generation fields (content creation)
         supplemental_content: str
         lesson_prompt: str
         quiz_prompt: str
         lesson_plan: str
         compiled_lesson: str
         quiz: str
         
         # Verification fields
         verification_report: str
         image_url: str
         
         # Quality fields
         evaluation_report: EvaluationReport
         
         # System fields
         execution_time: float
         error: str
         compilation_complete: bool
         
         # Output fields
         lesson_filename: str
         quiz_filename: str
     ```

2. **Nodes**
   Each node is a specialized function that processes specific aspects of the content generation:

   a. **Input Processing Nodes**
      - `intent_parser_node`: 
        - Extracts topic and grade level
        - Uses structured output (Pydantic model)
        - Updates: `topic`, `grade_level`

   b. **Knowledge Nodes**
      - `rag_agent_node`:
        - Searches source material
        - Validates topic coverage
        - Updates: `grounded_content`, `error`
        - Conditional: Determines if content generation should proceed

   c. **Creative Nodes**
      - `creative_assistant_node`:
        - Generates culturally relevant analogies
        - Uses Tavily search for context
        - Updates: `supplemental_content`
      
      - `enhanced_prompt_composer_node`:
        - Creates structured prompts
        - Combines RAG content with analogies
        - Updates: `lesson_prompt`, `quiz_prompt`

   d. **Generation Nodes**
      - `lesson_generator_node`:
        - Creates lesson plan
        - Enforces markdown formatting
        - Updates: `lesson_plan`
      
      - `quiz_generator_node`:
        - Creates assessment
        - Ensures grade-appropriate questions
        - Updates: `quiz`
      
      - `image_generator_node`:
        - Generates visual aids
        - Handles service fallbacks
        - Updates: `image_url`

   e. **Verification Nodes**
      - `hallucination_guard_node`:
        - Fact-checks against source
        - Ensures content accuracy
        - Updates: `verification_report`

   f. **Compilation Nodes**
      - `final_compiler_node`:
        - Combines all content
        - Adds images and formatting
        - Updates: `compiled_lesson`, `compilation_complete`
      
      - `evaluation_agent_node`:
        - Assesses content quality
        - Provides structured feedback
        - Updates: `evaluation_report`
      
      - `memory_agent_node`:
        - Handles persistence
        - Manages database and files
        - Updates: `lesson_filename`, `quiz_filename`

3. **Edges**
   The graph's flow is defined by directed edges between nodes:

   a. **Sequential Edges**
   ```python
   # Input to Knowledge
   builder.add_edge("Intent_Parser", "RAG_Agent")
   
   # Knowledge to Creative
   builder.add_edge("Creative_Assistant", "Enhanced_Prompt_Composer")
   
   # Verification to Compilation
   builder.add_edge("hallucination_guard", "Final_Compiler")
   ```

   b. **Parallel Edges**
   ```python
   # Parallel content generation
   builder.add_edge("Enhanced_Prompt_Composer", "Lesson_Generator")
   builder.add_edge("Enhanced_Prompt_Composer", "Quiz_Generator")
   builder.add_edge("Enhanced_Prompt_Composer", "Image_Generator")
   ```

   c. **Conditional Edges**
   ```python
   # RAG-based flow control
   builder.add_conditional_edges(
       "RAG_Agent",
       should_continue,  # Checks for errors
       {
           True: "Creative_Assistant",  # Continue if content found
           False: END  # Stop if no relevant content
       }
   )
   ```

4. **Flow Control**
   - Entry Point: `Intent_Parser`
   - Error Handling: Each node returns `{"error": message}` on failure
   - Parallel Processing: Content generation runs concurrently
   - State Updates: Nodes only update their designated keys
   - Completion: Flow ends at `Memory_Agent` or on error

### Development Guidelines

1. **Adding New Nodes**
   ```python
   def new_node(state: GraphState):
       """Template for new node."""
       print("---NODE: NEW NODE---")
       try:
           # Process state
           result = process_data(state)
           # Return only modified keys
           return {"new_key": result}
       except Exception as e:
           return {"error": str(e)}
   
   # Add to graph
   builder.add_node("New_Node", new_node)
   builder.add_edge("Previous_Node", "New_Node")
   ```

2. **State Management**
   - Always use TypedDict for type safety
   - Only update designated keys
   - Handle all error cases
   - Use clear, descriptive keys

3. **Testing**
   - Run tests: `python main.py --test`
   - Add node to `node_mapping` in tests
   - Update test state if adding new fields
   - Test error conditions

4. **Error Handling**
   ```python
   try:
       # Your node logic
       if error_condition:
           return {"error": "Specific error message"}
       return {"your_key": result}
   except Exception as e:
       return {"error": f"Error in node: {str(e)}"}
   ```

### Common Development Tasks

1. **Adding New Features**
   a. Update GraphState with new fields
   b. Create new node function
   c. Add node to graph builder
   d. Update test cases
   e. Add appropriate edges

2. **Modifying Flow**
   a. Identify affected nodes
   b. Update edge connections
   c. Add conditional logic if needed
   d. Update tests

3. **Adding New Content Types**
   a. Add new state fields
   b. Create generator node
   c. Add to Final_Compiler
   d. Update storage logic

4. **Improving Quality**
   a. Add verification steps
   b. Enhance evaluation criteria
   c. Update feedback mechanisms
   d. Add monitoring

### Best Practices

1. **Code Organization**
   - One responsibility per node
   - Clear state updates
   - Comprehensive error handling
   - Detailed logging

2. **Testing**
   - Test each node independently
   - Verify state updates
   - Mock external services
   - Check error handling

3. **Documentation**
   - Update GraphState comments
   - Document node responsibilities
   - Explain edge conditions
   - Keep diagrams current

4. **Performance**
   - Use parallel processing
   - Implement caching
   - Optimize RAG queries
   - Monitor execution time 