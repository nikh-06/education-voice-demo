from typing import Dict, Any
import unittest
from unittest.mock import patch, MagicMock
from main import (
    GraphState,
    intent_parser_node,
    rag_agent_node,
    creative_assistant_node,
    enhanced_prompt_composer_node,
    lesson_generator_node,
    quiz_generator_node,
    image_generator_node,
    hallucination_guard_node,
    final_compiler_node,
    evaluation_agent_node,
    memory_agent_node,
    listen_for_voice_command,
    graph
)
import speech_recognition as sr

def create_test_state() -> Dict[str, Any]:
    """Creates a test state with all required fields."""
    return {
        # Required input fields
        "user_request": "Test request",
        "topic": "Test topic",
        "grade_level": "5th grade",
        "retriever": None,  # Would be a real retriever object in production
        
        # Content generation fields
        "grounded_content": "Test grounded content from source material",
        "supplemental_content": "Test analogy and supplemental content",
        "lesson_prompt": "Test lesson generation prompt",
        "quiz_prompt": "Test quiz generation prompt",
        
        # Generated content fields
        "lesson_plan": "# Test Lesson Plan\n## Objectives\n- Test objective 1\n- Test objective 2",
        "compiled_lesson": None,  # Will be populated by Final_Compiler
        "quiz": "# Test Quiz\n1. Test question 1\n2. Test question 2",
        "verification_report": "All claims verified.",
        "image_url": "test_image.png",
        
        # Evaluation fields
        "evaluation_report": None,  # Will be populated by Evaluation_Agent
        
        # Metadata fields
        "execution_time": 0.0,
        "error": None,
        
        # Memory and output fields
        "lesson_filename": None,  # Will be populated by save_outputs_node
        "quiz_filename": None,    # Will be populated by save_outputs_node
        
        # Control flow fields
        "compilation_complete": False  # Important for Final_Compiler and subsequent nodes
    }

def validate_state_fields(test_state: Dict[str, Any]) -> bool:
    """Validates that the test state contains all required fields."""
    print("\nValidating test state fields...")
    graph_state_fields = set(GraphState.__annotations__.keys())
    test_state_fields = set(test_state.keys())
    
    # Check for missing fields
    missing_fields = graph_state_fields - test_state_fields
    if missing_fields:
        print(f"❌ Missing fields in test_state: {missing_fields}")
        return False
        
    # Check for extra fields
    extra_fields = test_state_fields - graph_state_fields
    if extra_fields:
        print(f"⚠️ Extra fields in test_state (not in GraphState): {extra_fields}")
    
    print("✅ Test state validation complete")
    return True

def test_node_outputs() -> bool:
    """Tests each node's output to ensure it only updates its expected keys."""
    print("\nAnalyzing node outputs...")
    
    # Map nodes to their output keys and functions
    node_mapping = {
        "Intent_Parser": {
            "function": intent_parser_node,
            "expected_outputs": ["topic", "grade_level"]
        },
        "RAG_Agent": {
            "function": rag_agent_node,
            "expected_outputs": ["grounded_content", "error"]
        },
        "Creative_Assistant": {
            "function": creative_assistant_node,
            "expected_outputs": ["supplemental_content"]
        },
        "Enhanced_Prompt_Composer": {
            "function": enhanced_prompt_composer_node,
            "expected_outputs": ["lesson_prompt", "quiz_prompt"]
        },
        "Lesson_Generator": {
            "function": lesson_generator_node,
            "expected_outputs": ["lesson_plan"]
        },
        "Quiz_Generator": {
            "function": quiz_generator_node,
            "expected_outputs": ["quiz"]
        },
        "Image_Generator": {
            "function": image_generator_node,
            "expected_outputs": ["image_url"]
        },
        "hallucination_guard": {
            "function": hallucination_guard_node,
            "expected_outputs": ["verification_report"]
        },
        "Final_Compiler": {
            "function": final_compiler_node,
            "expected_outputs": ["compiled_lesson", "compilation_complete"]
        },
        "Evaluation_Agent": {
            "function": evaluation_agent_node,
            "expected_outputs": ["evaluation_report"]
        },
        "Memory_Agent": {
            "function": memory_agent_node,
            "expected_outputs": ["lesson_filename", "quiz_filename"]
        }
    }
    
    # Create test state
    test_state = create_test_state()
    
    # Test each node's function
    test_results = {}
    for node_name, node_info in node_mapping.items():
        try:
            # Skip RAG_Agent as it requires actual retriever
            if node_name == "RAG_Agent":
                print(f"ℹ️ Skipping {node_name} (requires retriever)")
                continue
                
            # Call the function with test state
            print(f"\nTesting {node_name}...")
            result = node_info["function"](test_state.copy())
            
            if not isinstance(result, dict):
                print(f"❌ {node_name} returned non-dict result: {result}")
                test_results[node_name] = False
                continue
                
            # Check if the node updates only its expected keys
            actual_updates = set(result.keys())
            unexpected_updates = actual_updates - set(node_info["expected_outputs"]) - {"error"}
            if unexpected_updates:
                print(f"⚠️ {node_name} updates unexpected keys: {unexpected_updates}")
                test_results[node_name] = False
            else:
                print(f"✅ {node_name} updates only expected keys: {actual_updates}")
                test_results[node_name] = True
                
        except Exception as e:
            print(f"❌ Error testing {node_name}: {str(e)}")
            test_results[node_name] = False
    
    # Print summary
    print("\nTest Summary:")
    passed = sum(1 for result in test_results.values() if result)
    total = len(test_results)
    print(f"Passed: {passed}/{total} nodes")
    
    for node_name, passed in test_results.items():
        status = "✅" if passed else "❌"
        print(f"{status} {node_name}")
    
    print("\n✅ Node output analysis complete")
    return all(test_results.values())

class TestVoiceInput(unittest.TestCase):
    """Test cases for voice input functionality."""
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    def test_voice_input_english(self, mock_mic, mock_recognizer):
        """Test voice input in English."""
        # Setup mock
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        mock_recognizer_instance.recognize_google.return_value = "a lesson on photosynthesis for 5th grade"
        
        # Test
        result = listen_for_voice_command("en-IN")
        
        # Verify
        self.assertEqual(result, "a lesson on photosynthesis for 5th grade")
        mock_recognizer_instance.recognize_google.assert_called_once()
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    def test_voice_input_hindi(self, mock_mic, mock_recognizer):
        """Test voice input in Hindi."""
        # Setup mock
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        mock_recognizer_instance.recognize_google.return_value = "पांचवी कक्षा के लिए प्रकाश संश्लेषण पर एक पाठ"
        
        # Test
        result = listen_for_voice_command("hi-IN")
        
        # Verify
        self.assertEqual(result, "पांचवी कक्षा के लिए प्रकाश संश्लेषण पर एक पाठ")
        mock_recognizer_instance.recognize_google.assert_called_once()
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    def test_voice_input_timeout(self, mock_mic, mock_recognizer):
        """Test voice input timeout."""
        # Setup mock
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        mock_recognizer_instance.listen.side_effect = sr.WaitTimeoutError()
        
        # Test
        result = listen_for_voice_command()
        
        # Verify
        self.assertIsNone(result)
    
    @patch('speech_recognition.Recognizer')
    @patch('speech_recognition.Microphone')
    def test_voice_input_unknown_value(self, mock_mic, mock_recognizer):
        """Test voice input with unrecognizable speech."""
        # Setup mock
        mock_recognizer_instance = MagicMock()
        mock_recognizer.return_value = mock_recognizer_instance
        mock_recognizer_instance.recognize_google.side_effect = sr.UnknownValueError()
        
        # Test
        result = listen_for_voice_command()
        
        # Verify
        self.assertIsNone(result)

def run_all_tests() -> bool:
    """Runs all tests including graph structure and voice input tests."""
    print("\n--- Running All Tests ---")
    
    # Run graph structure tests
    print("\n1. Testing Graph Structure")
    test_state = create_test_state()
    if not validate_state_fields(test_state):
        return False
        
    if not test_node_outputs():
        return False
    
    # Run voice input tests
    print("\n2. Testing Voice Input")
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestVoiceInput)
    test_result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    if not test_result.wasSuccessful():
        print("\n❌ Voice input tests failed")
        return False
    
    print("\n✅ All tests passed")
    return True

if __name__ == "__main__":
    success = run_all_tests()
    if not success:
        print("\n❌ Tests failed")
        exit(1)
    print("\n✅ All tests passed successfully") 