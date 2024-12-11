from transformers import pipeline
import networkx as nx
import matplotlib.pyplot as plt

# Load pre-trained BERT model for NER
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", aggregation_strategy="simple")

def extract_entities_with_bert(text):
    """Extract entities using BERT."""
    ner_results = ner_pipeline(text)
    entities = list(set([result['word'] for result in ner_results]))
    return entities, ner_results

def extract_relationships(text, entities):
    """
    Simple relationship extraction logic.
    For more advanced relationships, use frameworks like spaCy or fine-tune a model.
    """
    relationships = []
    for entity in entities:
        # Example heuristic for relationships: word co-occurrence
        if " " + entity + " " in text:
            relationships.append((entity, "related_to", "context"))
    return relationships

# Example main function using BERT
def main_with_bert(pdf_file):
    # Step 1: Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_file)
    
    # Step 2: Extract entities using BERT
    print("Extracting entities with BERT...")
    entities, _ = extract_entities_with_bert(text)
    print("Entities:", entities)
    
    # Step 3: Extract relationships
    print("Extracting relationships...")
    relationships = extract_relationships(text, entities)
    print("Relationships:", relationships)
    
    # Step 4: Create and visualize the knowledge graph
    print("Creating knowledge graph...")
    create_knowledge_graph(entities, relationships)

# Example usage
if __name__ == "__main__":
    pdf_file_path = "example.pdf"  # Replace with the path to your PDF
    main_with_bert(pdf_file_path)
