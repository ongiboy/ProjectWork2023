def check_consecutive_repeats(values, repeat_count):
    current_count = 0
    previous_value = None
    
    for value in values:
        if value > 10:
            if value == previous_value:
                current_count += 1
            else:
                current_count = 1

            if current_count == repeat_count:
                #print(values)
                #print(value)
                return True

            previous_value = value

    return False
