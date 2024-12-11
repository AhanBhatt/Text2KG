import PyPDF2
import openai
import networkx as nx
import matplotlib.pyplot as plt

# Set your OpenAI API key
openai.api_key = "your_openai_api_key"

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text

def analyze_text_with_openai(text, model="gpt-4"):
    """Analyze text using OpenAI API to extract entities and relationships."""
    prompt = f"""
    Extract entities and their relationships from the following text:
    
    Text:
    {text}
    
    Provide the output in the format:
    Entities: [Entity1, Entity2, ...]
    Relationships: [(Entity1, Entity2, 'Relationship1'), (Entity3, Entity4, 'Relationship2'), ...]
    """
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def parse_openai_response(response):
    """Parse the response from OpenAI into entities and relationships."""
    lines = response.split("\n")
    entities_line = next(line for line in lines if line.startswith("Entities:"))
    relationships_line = next(line for line in lines if line.startswith("Relationships:"))
    
    entities = eval(entities_line.replace("Entities:", "").strip())
    relationships = eval(relationships_line.replace("Relationships:", "").strip())
    
    return entities, relationships

def create_knowledge_graph(entities, relationships):
    """Create and visualize a knowledge graph."""
    graph = nx.DiGraph()
    
    # Add nodes (entities)
    graph.add_nodes_from(entities)
    
    # Add edges (relationships)
    for entity1, entity2, relationship in relationships:
        graph.add_edge(entity1, entity2, label=relationship)
    
    # Draw the graph
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(12, 8))
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", node_size=3000, font_size=10)
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')
    plt.title("Knowledge Graph")
    plt.show()

def main(pdf_file):
    # Step 1: Extract text from PDF
    print("Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_file)
    
    # Step 2: Analyze text with OpenAI
    print("Analyzing text with OpenAI...")
    response = analyze_text_with_openai(text)
    print("OpenAI Response:\n", response)
    
    # Step 3: Parse the response
    print("Parsing response...")
    entities, relationships = parse_openai_response(response)
    print("Entities:", entities)
    print("Relationships:", relationships)
    
    # Step 4: Create and visualize the knowledge graph
    print("Creating knowledge graph...")
    create_knowledge_graph(entities, relationships)

# Example usage
if __name__ == "__main__":
    pdf_file_path = "example.pdf"  # Replace with the path to your PDF
    main(pdf_file_path)
