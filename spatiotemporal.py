import random
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import folium
from folium import plugins
import pandas as pd
from openai import OpenAI
import json
import time
from key import mykey

client = OpenAI(api_key=mykey)


def load_travel_times():
    try:
        df = pd.read_csv('min_travel_times.csv')
        travel_times = {}
        for _, row in df.iterrows():
            travel_times[(row['from_city'], row['to_city'])] = row['min_travel_time_seconds']
        return travel_times
    except Exception as e:
        print(f"Error loading travel times: {e}")
        return {}


places_data = {
    "Mumbai": {"lat": 19.0760, "lon": 72.8777},
    "Delhi": {"lat": 28.7041, "lon": 77.1025},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Goa": {"lat": 15.2993, "lon": 74.1240},
    "Jaipur": {"lat": 26.9124, "lon": 75.7873},
    "Kerala": {"lat": 10.8505, "lon": 76.2711},
    "Shimla": {"lat": 31.1048, "lon": 77.1734},
    "Darjeeling": {"lat": 27.0360, "lon": 88.2627},
    "Varanasi": {"lat": 25.3176, "lon": 82.9739},
    "Agra": {"lat": 27.1767, "lon": 78.0081},
    "Udaipur": {"lat": 24.5854, "lon": 73.7125},
    "Rishikesh": {"lat": 30.0869, "lon": 78.2676},
    "Mysore": {"lat": 12.2958, "lon": 76.6394},
    "Pondicherry": {"lat": 11.9139, "lon": 79.8145},
    "Amritsar": {"lat": 31.6340, "lon": 74.8723}
}

places = list(places_data.keys())


people = [
    "Rahul Sharma", "Priya Patel", "Amit Kumar", "Neha Gupta",
    "Vikram Singh", "Ananya Reddy", "Rohan Mehta", "Sneha Joshi",
    "Arjun Malhotra", "Ishaan Khanna"
]


min_travel_times = load_travel_times()

def get_min_travel_time(from_place, to_place):
    
    if (from_place, to_place) in min_travel_times:
        return min_travel_times[(from_place, to_place)]
    elif (to_place, from_place) in min_travel_times:
        return min_travel_times[(to_place, from_place)]
    else:
        
        return 14400  

def validate_edge(time_A, time_B, place_A, place_B):
    
    if time_A[1] > time_B[0]:
        return False
    
    
    min_travel = get_min_travel_time(place_A, place_B)
    if time_B[0] - time_A[1] < min_travel:
        return False
    
    return True

def generate_itinerary_with_llm(person, num_destinations=4):
    prompt1 = f"""Generate a travel itinerary for {person} visiting {num_destinations} destinations in India.
    Available destinations: {', '.join(places)}
    
    Return ONLY a JSON object with the following structure:
    {{
        "itinerary": [
            {{
                "place": "city_name",
                "arrival_time": "YYYY-MM-DD HH:MM",
                "departure_time": "YYYY-MM-DD HH:MM"
            }},
            ...
        ]
    }}
    
    Important: 
    1. Use 24-hour format but never use 24:00 (use 00:00 instead)
    2. Use realistic dates within the next 30 days
    3. Return ONLY the JSON object, no additional text or explanation
    """

    prompt2 = f"""Consider planes that are superfast and can travel 10 times faster than the current planes. Generate a travel itinerary for {person} visiting {num_destinations} destinations in India.
    Available destinations: {', '.join(places)}
    
    Return ONLY a JSON object with the following structure:
    {{
        "itinerary": [
            {{
                "place": "city_name",
                "arrival_time": "YYYY-MM-DD HH:MM",
                "departure_time": "YYYY-MM-DD HH:MM"
            }},
            ...
        ]
    }}
    
    Important: 
    1. Use 24-hour format but never use 24:00 (use 00:00 instead)
    2. Use realistic dates within the next 30 days
    3. Return ONLY the JSON object, no additional text or explanation
    """
    
    try:
        
        response1 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a travel itinerary planner that generates travel plans. Return only valid JSON."},
                {"role": "user", "content": prompt1}
            ],
            temperature=0.7
        )
        
        
        response2 = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a travel itinerary planner that generates travel plans. Return only valid JSON."},
                {"role": "user", "content": prompt2}
            ],
            temperature=0.7
        )
        
        
        content1 = response1.choices[0].message.content.strip()
        start_idx1 = content1.find('{')
        end_idx1 = content1.rfind('}') + 1
        if start_idx1 == -1 or end_idx1 == 0:
            print(f"Error: No JSON object found in response1: {content1}")
            itinerary1 = None
        else:
            json_str1 = content1[start_idx1:end_idx1]
            itinerary_json1 = json.loads(json_str1)
            if "itinerary" not in itinerary_json1 or not isinstance(itinerary_json1["itinerary"], list):
                print(f"Error: Invalid itinerary structure in response1: {itinerary_json1}")
                itinerary1 = None
            else:
                itinerary1 = itinerary_json1["itinerary"]
        
        
        content2 = response2.choices[0].message.content.strip()
        start_idx2 = content2.find('{')
        end_idx2 = content2.rfind('}') + 1
        if start_idx2 == -1 or end_idx2 == 0:
            print(f"Error: No JSON object found in response2: {content2}")
            itinerary2 = None
        else:
            json_str2 = content2[start_idx2:end_idx2]
            itinerary_json2 = json.loads(json_str2)
            if "itinerary" not in itinerary_json2 or not isinstance(itinerary_json2["itinerary"], list):
                print(f"Error: Invalid itinerary structure in response2: {itinerary_json2}")
                itinerary2 = None
            else:
                itinerary2 = itinerary_json2["itinerary"]
        
        return itinerary1, itinerary2
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None, None
    except Exception as e:
        print(f"Error generating itinerary: {e}")
        return None, None

def convert_itinerary_to_journey(itinerary):
    journey_events = []
    for stop in itinerary:
        place = stop["place"]
        
        arrival_time = stop["arrival_time"]
        departure_time = stop["departure_time"]
        
        if "24:00" in arrival_time:
            date = datetime.strptime(arrival_time, "%Y-%m-%d %H:%M")
            date = date + timedelta(days=1)
            arrival_time = date.strftime("%Y-%m-%d 00:00")
            
        if "24:00" in departure_time:
            date = datetime.strptime(departure_time, "%Y-%m-%d %H:%M")
            date = date + timedelta(days=1)
            departure_time = date.strftime("%Y-%m-%d 00:00")
        
        try:
            arrival = int(datetime.strptime(arrival_time, "%Y-%m-%d %H:%M").timestamp())
            departure = int(datetime.strptime(departure_time, "%Y-%m-%d %H:%M").timestamp())
            journey_events.append((place, (arrival, departure)))
        except ValueError as e:
            print(f"Error converting time: {e}")
            print(f"Problematic times - Arrival: {arrival_time}, Departure: {departure_time}")
            return None
            
    return journey_events

def validate_journey(journey_events):
    print("\nValidating journey...")
    for i in range(len(journey_events) - 1):
        place_A, time_A = journey_events[i]
        place_B, time_B = journey_events[i + 1]
        
        print(f"\nChecking segment: {place_A} -> {place_B}")
        print(f"Departure from {place_A}: {format_timestamp(time_A[1])}")
        print(f"Arrival at {place_B}: {format_timestamp(time_B[0])}")
        
        if not validate_edge(time_A, time_B, place_A, place_B):
            travel_time = time_B[0] - time_A[1]
            min_required = get_min_travel_time(place_A, place_B)
            print(f"INCONSISTENCY FOUND:")
            print(f"Travel time: {travel_time/3600:.1f} hours")
            print(f"Minimum required: {min_required/3600:.1f} hours")
            return False
        
        print("Segment validated successfully")
    return True

def generate_valid_journey(person, is_normal=True, max_attempts=3):
    """Generate and validate a journey.
    Args:
        person: Person name
        is_normal: If True, generates normal itinerary (prompt1), else superfast (prompt2)
        max_attempts: Maximum number of attempts to generate valid itinerary
    """
    itinerary_type = "normal" if is_normal else "superfast"
    print(f"\nGenerating {itinerary_type} itinerary...")
    journey = None
    
    for attempt in range(max_attempts):
        print(f"\n{itinerary_type.title()} Itinerary - Attempt {attempt + 1} of {max_attempts}")
        
        
        if is_normal:
            itinerary, _ = generate_itinerary_with_llm(person)
        else:
            _, itinerary = generate_itinerary_with_llm(person)
            
        if not itinerary:
            print(f"Failed to generate {itinerary_type} itinerary")
            continue
            
        print(f"\nGenerated {itinerary_type.title()} Itinerary:")
        for stop in itinerary:
            print(f"{stop['place']}: Arrive {stop['arrival_time']} - Depart {stop['departure_time']}")
            
        journey_events = convert_itinerary_to_journey(itinerary)
        valid = validate_journey(journey_events)
        
        if valid:
            print(f"\nValid {itinerary_type} itinerary found!")
            journey = journey_events
            break
            
        
        inconsistency_info = []
        for i in range(len(journey_events) - 1):
            place_A, time_A = journey_events[i]
            place_B, time_B = journey_events[i + 1]
            if not validate_edge(time_A, time_B, place_A, place_B):
                travel_time = time_B[0] - time_A[1]
                min_required = get_min_travel_time(place_A, place_B)
                inconsistency_info.append(
                    f"Min. required time to travel between {place_A} and {place_B} is {min_required/3600:.1f} hours "
                    f"but the itinerary allots only {travel_time/3600:.1f} hours which is not possible."
                )
        
        if inconsistency_info:
            print(f"\nInconsistencies found in {itinerary_type} itinerary:")
            for info in inconsistency_info:
                print(f"- {info}")
            
            
            inconsistency_prompt = f"""Previous itinerary had the following inconsistencies:
{chr(10).join(inconsistency_info)}
Please removing these inconsistencies by taking into account the minimum travel time between the cities and return a valid and reasonable itinerary.
"""
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a travel itinerary planner that generates travel plans. Return only valid JSON."},
                        {"role": "user", "content": inconsistency_prompt}
                    ],
                    temperature=0.7
                )
                
                
                content = response.choices[0].message.content.strip()
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    itinerary_json = json.loads(json_str)
                    if "itinerary" in itinerary_json and isinstance(itinerary_json["itinerary"], list):
                        
                        itinerary = itinerary_json["itinerary"]
                        print(f"\nGenerated new {itinerary_type} itinerary with inconsistency feedback:")
                        for stop in itinerary:
                            print(f"{stop['place']}: Arrive {stop['arrival_time']} - Depart {stop['departure_time']}")
            except Exception as e:
                print(f"Error generating new {itinerary_type} itinerary: {e}")
        
        print(f"\nAttempt {attempt + 1}: {itinerary_type} itinerary had inconsistencies. Retrying...")
        time.sleep(1)
    
    if not journey:
        print(f"Failed to generate valid {itinerary_type} itinerary for {person} after {max_attempts} attempts")
    
    return journey

def format_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M')

def validate_graph(G):
    inconsistent_edges = []
    
    for u, v, data in G.edges(data=True):
        time_A = data.get("time_A")
        time_B = data.get("time_B")
        if not validate_edge(time_A, time_B, u, v):
            inconsistent_edges.append((u, v, data))
    
    try:
        cycle = nx.find_cycle(G, orientation='original')
        cycle_detected = True
    except nx.NetworkXNoCycle:
        cycle_detected = False
    
    return inconsistent_edges, cycle_detected

def generate_random_journeys(people, places):
    """Generate random journeys for all people with a mix of consistent and inconsistent events."""
    journeys_by_person = {}
    G = nx.MultiDiGraph()
    G.add_nodes_from(places)
    
    for person in people:
        
        journey_length = random.randint(3, 6)
        
        
        journey_places = random.choices(places, k=journey_length)
        
        
        journey_events = []
        
        
        first_event = generate_event(prev_end=None)
        journey_events.append((journey_places[0], first_event))
        prev_end = first_event[1]
        
        
        for i in range(1, journey_length):
            if random.random() < 0.4:
                event_time = generate_event(prev_end, consistent=True)
            else:
                event_time = generate_event(prev_end, consistent=False)
            journey_events.append((journey_places[i], event_time))
            prev_end = event_time[1]
        
        
        journeys_by_person[person] = journey_events
        
        
        for i in range(len(journey_events) - 1):
            place_A, time_A = journey_events[i]
            place_B, time_B = journey_events[i+1]
            G.add_edge(place_A, place_B, person=person, time_A=time_A, time_B=time_B)
    
    return journeys_by_person, G

def generate_event(prev_end=None, consistent=True):
    """Generate a random event with arrival and departure times.
    
    Args:
        prev_end: Previous event's end time (timestamp) or None for first event
        consistent: Whether to generate consistent timing with minimum travel times
    
    Returns:
        Tuple of (arrival_time, departure_time) as timestamps
    """
    
    if prev_end is None:
        base_time = datetime.now()
    else:
        base_time = datetime.fromtimestamp(prev_end)
    
    
    if prev_end is None:
        
        arrival_time = base_time + timedelta(days=random.randint(0, 7))
    else:
        if consistent:
            
            arrival_time = base_time + timedelta(seconds=3600)  
        else:
            
            arrival_time = base_time + timedelta(seconds=random.randint(0, 3600))
    
    
    stay_duration = random.randint(24, 72) * 3600  
    departure_time = arrival_time + timedelta(seconds=stay_duration)
    
    
    return (int(arrival_time.timestamp()), int(departure_time.timestamp()))

def main():
    
    print("\n=== RANDOM ITINERARIES ===")
    print("\nGenerating random itineraries for all people...")
    random_journeys_by_person, G_random = generate_random_journeys(people, places)
    
    
    print("\n--- Random Journeys by Person ---")
    for person in sorted(random_journeys_by_person.keys()):
        journey = random_journeys_by_person[person]
        journey_str = " -> ".join(
            [f"{place}({format_timestamp(time[0])}-{format_timestamp(time[1])})" for place, time in journey]
        )
        print(f"{person}: {journey_str}")
    
    
    inconsistencies_random, cycle_found_random = validate_graph(G_random)
    print("\n--- Validator Report for Random Itineraries ---")
    if inconsistencies_random:
        print("Inconsistent edges found:")
        for u, v, data in inconsistencies_random:
            time_A = data['time_A']
            time_B = data['time_B']
            travel_time = time_B[0] - time_A[1]
            min_required = get_min_travel_time(u, v)
            
            print(f"\n{u} -> {v} | Person: {data['person']}")
            print(f"Departure: {format_timestamp(time_A[1])}")
            print(f"Arrival: {format_timestamp(time_B[0])}")
            print(f"Travel time: {travel_time/3600:.1f} hours (Minimum required: {min_required/3600:.1f} hours)")
            
            
            journey = random_journeys_by_person[data['person']]
            print("\nFull journey with inconsistency:")
            for place, (arrival, departure) in journey:
                print(f"{place}: Arrive {format_timestamp(arrival)} - Depart {format_timestamp(departure)}")
    else:
        print("No inconsistent edges found.")

    if cycle_found_random:
        print("\nCycle detected in the random itineraries graph!")
    else:
        print("\nNo cycle detected in the random itineraries graph.")
    
    
    print("\nCreating map visualization for random itineraries...")
    create_map_visualization(G_random, random_journeys_by_person, 'random_itineraries_map.html')
    
    
    print("\n\n=== NORMAL LLM ITINERARY ===")
    person = people[0]
    print(f"\nGenerating normal LLM itinerary for {person}...")
    journey_events1 = generate_valid_journey(person, is_normal=True)
    
    if journey_events1:
        
        G_normal = nx.MultiDiGraph()
        G_normal.add_nodes_from(places)
        
        
        for i in range(len(journey_events1) - 1):
            place_A, time_A = journey_events1[i]
            place_B, time_B = journey_events1[i + 1]
            G_normal.add_edge(place_A, place_B, person=person, time_A=time_A, time_B=time_B)
        
        
        print("\n--- Normal LLM Journey ---")
        journey_str1 = " -> ".join(
            [f"{place}({format_timestamp(time[0])}-{format_timestamp(time[1])})" for place, time in journey_events1]
        )
        print(f"{person}: {journey_str1}")
        
        
        inconsistencies_normal, cycle_found_normal = validate_graph(G_normal)
        print("\n--- Validator Report for Normal LLM Itinerary ---")
        if inconsistencies_normal:
            print("Inconsistent edges found:")
            for u, v, data in inconsistencies_normal:
                time_A = data['time_A']
                time_B = data['time_B']
                travel_time = time_B[0] - time_A[1]
                min_required = get_min_travel_time(u, v)
                
                print(f"\n{u} -> {v} | Person: {data['person']}")
                print(f"Departure: {format_timestamp(time_A[1])}")
                print(f"Arrival: {format_timestamp(time_B[0])}")
                print(f"Travel time: {travel_time/3600:.1f} hours (Minimum required: {min_required/3600:.1f} hours)")
                
                
                print("\nFull journey with inconsistency:")
                for place, (arrival, departure) in journey_events1:
                    print(f"{place}: Arrive {format_timestamp(arrival)} - Depart {format_timestamp(departure)}")
        else:
            print("No inconsistent edges found.")

        if cycle_found_normal:
            print("\nCycle detected in the normal itinerary graph!")
        else:
            print("\nNo cycle detected in the normal itinerary graph.")
        
        
        print("\nCreating map visualization for normal LLM itinerary...")
        create_map_visualization(G_normal, {person: (journey_events1, None)}, 'LLM_itineraries_travel_map1.html')
    
    
    print("\n\n=== SUPERFAST LLM ITINERARY ===")
    print(f"\nGenerating superfast LLM itinerary for {person}...")
    journey_events2 = generate_valid_journey(person, is_normal=False)
    
    if journey_events2:
        
        G_superfast = nx.MultiDiGraph()
        G_superfast.add_nodes_from(places)
        
        
        for i in range(len(journey_events2) - 1):
            place_A, time_A = journey_events2[i]
            place_B, time_B = journey_events2[i + 1]
            G_superfast.add_edge(place_A, place_B, person=person, time_A=time_A, time_B=time_B)
        
        
        print("\n--- Superfast LLM Journey ---")
        journey_str2 = " -> ".join(
            [f"{place}({format_timestamp(time[0])}-{format_timestamp(time[1])})" for place, time in journey_events2]
        )
        print(f"{person}: {journey_str2}")
        
        
        inconsistencies_superfast, cycle_found_superfast = validate_graph(G_superfast)
        print("\n--- Validator Report for Superfast LLM Itinerary ---")
        if inconsistencies_superfast:
            print("Inconsistent edges found:")
            for u, v, data in inconsistencies_superfast:
                time_A = data['time_A']
                time_B = data['time_B']
                travel_time = time_B[0] - time_A[1]
                min_required = get_min_travel_time(u, v)
                
                print(f"\n{u} -> {v} | Person: {data['person']}")
                print(f"Departure: {format_timestamp(time_A[1])}")
                print(f"Arrival: {format_timestamp(time_B[0])}")
                print(f"Travel time: {travel_time/3600:.1f} hours (Minimum required: {min_required/3600:.1f} hours)")
                
                
                print("\nFull journey with inconsistency:")
                for place, (arrival, departure) in journey_events2:
                    print(f"{place}: Arrive {format_timestamp(arrival)} - Depart {format_timestamp(departure)}")
        else:
            print("No inconsistent edges found.")

        if cycle_found_superfast:
            print("\nCycle detected in the superfast itinerary graph!")
        else:
            print("\nNo cycle detected in the superfast itinerary graph.")
    
    
    print("\nCreating map visualization for superfast LLM itinerary...")
    create_map_visualization(G_superfast if journey_events2 else nx.MultiDiGraph(), 
                           {person: (None, journey_events2)} if journey_events2 else {}, 
                           'LLM_itineraries_travel_map2.html')

def create_map_visualization(G, journeys_by_person, output_file):
    
    india_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)
    
    
    for place in places:
        
        visits = []
        for person, journeys in journeys_by_person.items():
            
            if isinstance(journeys, tuple):
                journey1, journey2 = journeys
                if journey1:
                    for visit_place, (arrival, departure) in journey1:
                        if visit_place == place:
                            visits.append(f"{person} (Normal):<br>"
                                       f"Arrival: {format_timestamp(arrival)}<br>"
                                       f"Departure: {format_timestamp(departure)}")
                if journey2:
                    for visit_place, (arrival, departure) in journey2:
                        if visit_place == place:
                            visits.append(f"{person} (Fast Travel):<br>"
                                       f"Arrival: {format_timestamp(arrival)}<br>"
                                       f"Departure: {format_timestamp(departure)}")
            else:
                
                for visit_place, (arrival, departure) in journeys:
                    if visit_place == place:
                        visits.append(f"{person}:<br>"
                                   f"Arrival: {format_timestamp(arrival)}<br>"
                                   f"Departure: {format_timestamp(departure)}")
        
        popup_content = f"<b>{place}</b><br>"
        if visits:
            popup_content += "<br>" + "<br><br>".join(visits)
        
        folium.Marker(
            location=[places_data[place]["lat"], places_data[place]["lon"]],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(india_map)
    
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
    
    for i, (person, journeys) in enumerate(journeys_by_person.items()):
        
        if isinstance(journeys, tuple):
            journey1, journey2 = journeys
            if journey1:
                
                color = colors[i % len(colors)]
                for j in range(len(journey1) - 1):
                    place_A, time_A = journey1[j]
                    place_B, time_B = journey1[j + 1]
                    
                    
                    start_coords = [places_data[place_A]["lat"], places_data[place_A]["lon"]]
                    end_coords = [places_data[place_B]["lat"], places_data[place_B]["lon"]]
                    
                    
                    mid_lat = (start_coords[0] + end_coords[0]) / 2
                    mid_lon = (start_coords[1] + end_coords[1]) / 2
                    
                    
                    travel_time = (time_B[0] - time_A[1]) / 3600
                    
                    
                    folium.PolyLine(
                        locations=[start_coords, end_coords],
                        color=color,
                        weight=2,
                        opacity=0.8,
                        popup=folium.Popup(
                            f"<b>{person}'s Normal Journey</b><br>"
                            f"From: {place_A}<br>"
                            f"To: {place_B}<br>"
                            f"Departure: {format_timestamp(time_A[1])}<br>"
                            f"Arrival: {format_timestamp(time_B[0])}<br>"
                            f"Travel Time: {travel_time:.1f} hours<br>"
                            f"Minimum Required Time: {get_min_travel_time(place_A, place_B)/3600:.1f} hours",
                            max_width=300
                        )
                    ).add_to(india_map)
                    
                    
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10pt; color: {color};">{travel_time:.1f}h</div>'
                        ),
                        popup=folium.Popup(
                            f"Travel Time: {travel_time:.1f} hours<br>"
                            f"Minimum Required: {get_min_travel_time(place_A, place_B)/3600:.1f} hours",
                            max_width=200
                        )
                    ).add_to(india_map)
            
            if journey2:
                
                color = colors[(i + 1) % len(colors)]  
                for j in range(len(journey2) - 1):
                    place_A, time_A = journey2[j]
                    place_B, time_B = journey2[j + 1]
                    
                    
                    start_coords = [places_data[place_A]["lat"], places_data[place_A]["lon"]]
                    end_coords = [places_data[place_B]["lat"], places_data[place_B]["lon"]]
                    
                    
                    mid_lat = (start_coords[0] + end_coords[0]) / 2
                    mid_lon = (start_coords[1] + end_coords[1]) / 2
                    
                    
                    travel_time = (time_B[0] - time_A[1]) / 3600
                    
                    
                    folium.PolyLine(
                        locations=[start_coords, end_coords],
                        color=color,
                        weight=2,
                        opacity=0.8,
                        dash_array='5, 10',  
                        popup=folium.Popup(
                            f"<b>{person}'s Fast Travel Journey</b><br>"
                            f"From: {place_A}<br>"
                            f"To: {place_B}<br>"
                            f"Departure: {format_timestamp(time_A[1])}<br>"
                            f"Arrival: {format_timestamp(time_B[0])}<br>"
                            f"Travel Time: {travel_time:.1f} hours<br>"
                            f"Minimum Required Time: {get_min_travel_time(place_A, place_B)/3600:.1f} hours",
                            max_width=300
                        )
                    ).add_to(india_map)
                    
                    
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10pt; color: {color};">{travel_time:.1f}h</div>'
                        ),
                        popup=folium.Popup(
                            f"Travel Time: {travel_time:.1f} hours<br>"
                            f"Minimum Required: {get_min_travel_time(place_A, place_B)/3600:.1f} hours",
                            max_width=200
                        )
                    ).add_to(india_map)
        else:
            
            color = colors[i % len(colors)]
            for j in range(len(journeys) - 1):
                place_A, time_A = journeys[j]
                place_B, time_B = journeys[j + 1]
                
                
                start_coords = [places_data[place_A]["lat"], places_data[place_A]["lon"]]
                end_coords = [places_data[place_B]["lat"], places_data[place_B]["lon"]]
                
                
                mid_lat = (start_coords[0] + end_coords[0]) / 2
                mid_lon = (start_coords[1] + end_coords[1]) / 2
                
                
                travel_time = (time_B[0] - time_A[1]) / 3600
                
                
                folium.PolyLine(
                    locations=[start_coords, end_coords],
                    color=color,
                    weight=2,
                    opacity=0.8,
                    popup=folium.Popup(
                        f"<b>{person}'s Journey</b><br>"
                        f"From: {place_A}<br>"
                        f"To: {place_B}<br>"
                        f"Departure: {format_timestamp(time_A[1])}<br>"
                        f"Arrival: {format_timestamp(time_B[0])}<br>"
                        f"Travel Time: {travel_time:.1f} hours<br>"
                        f"Minimum Required Time: {get_min_travel_time(place_A, place_B)/3600:.1f} hours",
                        max_width=300
                    )
                ).add_to(india_map)
                
                
                folium.Marker(
                    location=[mid_lat, mid_lon],
                    icon=folium.DivIcon(
                        html=f'<div style="font-size: 10pt; color: {color};">{travel_time:.1f}h</div>'
                    ),
                    popup=folium.Popup(
                        f"Travel Time: {travel_time:.1f} hours<br>"
                        f"Minimum Required: {get_min_travel_time(place_A, place_B)/3600:.1f} hours",
                        max_width=200
                    )
                ).add_to(india_map)
    
    
    folium.LayerControl().add_to(india_map)
    
    
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; width: 150px; height: 120px; 
                border:2px solid grey; z-index:9999; 
                background-color:white; padding: 10px;
                font-size: 14px;">
     <p><b>Legend</b></p>
     <p>Solid lines: Normal Itinerary</p>
     <p>Dashed lines: Fast Travel Itinerary</p>
    </div>
    '''
    india_map.get_root().html.add_child(folium.Element(legend_html))
    
    
    india_map.save(output_file)
    print(f"\nMap visualization has been saved as '{output_file}'")
    print("Open this file in a web browser to view the interactive map.")

if __name__ == "__main__":
    main() 