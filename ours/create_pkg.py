import pandas as pd
from py2neo import Graph, Node, Relationship


data_set_df = pd.read_excel("basic_data.xlsx", engine='openpyxl')
neo4j_graph = Graph("bolt://localhost:7687", auth=("neo4j", "12345678"))
neo4j_graph.run("MATCH (n) DETACH DELETE n")

for _, row in data_set_df.iterrows():
    post_label = row["违规情况"]
    post_node = Node(
        post_label,
        帖子ID=str(row["帖子ID"]),
        标题=str(row["标题"]),
        正文文本=str(row["正文文本"]),
        回复贴1=str(row["回复贴1"]),
        回复贴2=str(row["回复贴2"]),
        所在吧名=str(row["所在吧名"]),
        发布时间=str(row["发布时间"]),
        回复贴=str(row["回复贴"]),
        情感分类1=str(row["情感分类1"]),
        情感分类2=str(row["情感分类2"])
    )
    neo4j_graph.create(post_node)

    user_name = str(row["用户名"])
    user_node = neo4j_graph.evaluate("MATCH (u:用户 {用户名: $user_name}) RETURN u", user_name=user_name)
    if not user_node:
        user_node = Node(
            "用户",
            用户名=str(row["用户名"]),
            吧龄_年=str(row["吧龄_年"]),
            发帖量=str(row["发帖量"]),
            IP属地=str(row["IP属地"]),
            他关注的人=str(row["他关注的人"]),
            关注他的人=str(row["关注他的人"])
        )
        neo4j_graph.create(user_node)

    publish_relationship = Relationship(user_node, "发布", post_node, 关系类型="发布")
    neo4j_graph.create(publish_relationship)


    other_posts = neo4j_graph.run(
        "MATCH (u:用户)-[:发布]->(p) WHERE u.用户名 = $user_name AND p.帖子ID <> $post_id RETURN p",
        user_name=user_name,
        post_id=str(row["帖子ID"])
    ).data()

    for other_post in other_posts:
        other_post_node = other_post["p"]
        same_user_relationship = Relationship(post_node, "同一用户", other_post_node, 关系类型="同一用户")
        neo4j_graph.create(same_user_relationship)

    same_bar_posts = neo4j_graph.run(
        "MATCH (p {所在吧名: $bar_name}) WHERE p.帖子ID <> $post_id RETURN p",
        bar_name=str(row["所在吧名"]),
        post_id=str(row["帖子ID"])
    ).data()

    for same_bar_post in same_bar_posts:
        same_bar_post_node = same_bar_post["p"]

        existing_relationship = neo4j_graph.evaluate(
            "MATCH (p1)-[r:同一贴吧]-(p2) WHERE p1.帖子ID = $post_id AND p2.帖子ID = $same_bar_post_id RETURN r",
            post_id=str(row["帖子ID"]),
            same_bar_post_id=str(same_bar_post_node["帖子ID"])
        )

        if not existing_relationship:
            same_bar_relationship = Relationship(post_node, "同一贴吧", same_bar_post_node, 关系类型="同一贴吧")
            neo4j_graph.create(same_bar_relationship)


user_relation_df = pd.read_excel("user_relation.xlsx", engine='openpyxl')

for _, row in user_relation_df.iterrows():
    user_1_name = row["user_1"]
    user_2_name = row["user_2"]
    relation_type = row["relation"]

    user_1_node = neo4j_graph.evaluate("MATCH (u:用户 {用户名: $user_1_name}) RETURN u", user_1_name=user_1_name)
    user_2_node = neo4j_graph.evaluate("MATCH (u:用户 {用户名: $user_2_name}) RETURN u", user_2_name=user_2_name)
    if not user_1_node or not user_2_node:
        continue

    if relation_type == "关注":
        relationship = Relationship(user_1_node, "关注", user_2_node, 关系类型="关注")
    elif relation_type == "回复":
        relationship = Relationship(user_1_node, "回复", user_2_node, 关系类型="回复")
    else:
        continue

    neo4j_graph.create(relationship)

    user_1_posts = neo4j_graph.run(
        "MATCH (u:用户)-[:发布]->(p) WHERE u.用户名 = $user_1_name RETURN p",
        user_1_name=user_1_name
    ).data()

    user_2_posts = neo4j_graph.run(
        "MATCH (u:用户)-[:发布]->(p) WHERE u.用户名 = $user_2_name RETURN p",
        user_2_name=user_2_name
    ).data()

    for post_1 in user_1_posts:
        post_1_node = post_1["p"]
        for post_2 in user_2_posts:
            post_2_node = post_2["p"]

            existing_relationship = neo4j_graph.evaluate(
                "MATCH (p1)-[r:相关用户]->(p2) WHERE p1.帖子ID = $post_1_id AND p2.帖子ID = $post_2_id RETURN r",
                post_1_id=str(post_1_node["帖子ID"]),
                post_2_id=str(post_2_node["帖子ID"])
            )

            if not existing_relationship:
                related_user_relationship = Relationship(post_1_node, "相关用户", post_2_node, 关系类型="相关用户")
                neo4j_graph.create(related_user_relationship)


neo4j_graph.run("""
    MATCH (u:用户)
    DETACH DELETE u
""")

print("帖子知识图谱创建完成！")