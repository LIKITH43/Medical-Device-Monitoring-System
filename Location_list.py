# Define hospital locations (A–F) and regions programmatically
hospitals = ["Hospital A", "Hospital B", "Hospital C",
             "Hospital D", "Hospital E", "Hospital F"]

regions = ["Central Region", "East Region", "North Region",
           "South Region", "West Region"]

Location_list = [f"{h} - {r}" for h in hospitals for r in regions]
