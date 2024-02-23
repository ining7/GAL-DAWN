/**
 * @author lxrzlyr (1289539524@qq.com)
 * @date 2024-02-23
 *
 * @copyright Copyright (c) 2024
 */
#!/bin/bash

# 获取配置信息
username=$(git config user.name)
email=$(git config user.email)
date=$(date "+%Y-%m-%d")

# 输出版权信息，不包含文件名
echo "/**"
echo " * @author $username ($email)"
echo " * @date $date"
echo " *"
echo " * @copyright Copyright (c) $(date +%Y)"
echo " */"

# 输出原始文件内容
cat
